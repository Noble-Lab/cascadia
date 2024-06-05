from depthcharge.depthcharge.encoders import PeakEncoder, FloatEncoder
from depthcharge.depthcharge.transformers import SpectrumTransformerEncoder, PeptideTransformerDecoder
from typing import Any, Dict, Iterable, List, Tuple
import torch
import numpy as np
import pytorch_lightning as pl

class AugmentedPeakEncoder(torch.nn.Module):
    """Encode an augmented m/z, intensity,

    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_rt_wavelength : float, optional
        The minimum wavelength to use for m/z.
    max_rt_wavelength : float, optional
        The maximum wavelength to use for m/z.
    min_intensity_wavelength : float, optional
        The minimum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    max_intensity_wavelength : float, optional
        The maximum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    """

    def __init__(
        self,
        d_model: int,
        min_intensity_wavelength: float = 1e-6,
        max_intensity_wavelength: float = 1,
        min_rt_wavelength: float = 1e-6,
        max_rt_wavelength: float = 10
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__()
        self.d_model = d_model

        self.peak_encoder = PeakEncoder(
            d_model,
            min_intensity_wavelength,
            max_intensity_wavelength,
            learnable_wavelengths = False
        )

        self.rt_encoder = FloatEncoder(
            d_model,
            min_wavelength=min_rt_wavelength,
            max_wavelength=max_rt_wavelength
        )

        self.level_encoder = torch.nn.Embedding(3, d_model)

        self.combiner = torch.nn.Linear(3 * d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values, intensities, rts, and mslevel

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 4)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is a 4-tuple
            specifying the (m/z, intensity, retention time, ms level) for each peak.
            These are zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, d_model)
            The encoded features for the augmented mass spectra.
        """

        encoded = torch.cat(
            [
                self.peak_encoder(X),
                self.rt_encoder(X[:, :, 2]),
                self.level_encoder(X[:, :, 3].int())
            ],
            dim=2,
        )

        return self.combiner(encoded)

class AugmentedSpec2Pep(pl.LightningModule):
    def __init__(
            self,
            d_model,
            n_layers,
            rt_width,
            n_head,
            dropout,
            dim_feedforward,
            tokenizer,
            max_charge,
            lr,
            self.frag_weight = 20,
            self.lr_decay=1e-9
        ):

        super().__init__()

        self.peak_encoder = AugmentedPeakEncoder(
            d_model=d_model,
            max_rt_wavelength=2*rt_width
        )

        self.spectrum_encoder = SpectrumTransformerEncoder(
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            peak_encoder=self.peak_encoder
        )

        self.decoder = PeptideTransformerDecoder(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            n_tokens=tokenizer,
            max_charge=max_charge
        )

        self.prec_layer = torch.nn.Linear(d_model,1)
        self.frag_layer = torch.nn.Linear(d_model,2)

        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.fragCELoss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,self.frag_weight]))
        self.MSELoss = torch.nn.MSELoss()
        self.lr = lr

        self.tokenizer = tokenizer

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.spectrum_encoder(spectra)
        result =  self.decoder(sequences, precursors, *emb)
        spec_rep = emb[0][:,0]
        pred_prec = self.prec_layer(spec_rep)[:,0]
        pred_frag = self.frag_layer(emb[0])

        return result, pred_prec, pred_frag

    def training_step(self, batch):

        spectra, precursors, sequences, frag_labels = batch
        preds, pred_prec, pred_frags= self._forward_step(spectra, precursors, sequences)
        preds_seqs = torch.argmax(preds, dim=2)[:,:-1]
        preds = preds[:,:-1,:].reshape(-1, len(self.tokenizer) + 1)
        true_seqs = batch[2][:,:]

        loss = self.CELoss(preds,true_seqs.flatten())

        true_prec = batch[1][:,0]/1000
        mse_loss = self.MSELoss(pred_prec, true_prec)

        pred_frags = pred_frags[:,1:,:].reshape(-1, 2)
        frag_loss = self.fragCELoss(pred_frags, frag_labels)

        correct_aas = torch.logical_or(preds_seqs == true_seqs, true_seqs == 0)
        aa_acc = torch.mean(correct_aas.float())

        correct_peps = torch.mean(correct_aas.float(), dim=1) == 1
        pep_acc = torch.mean(correct_peps.float())

        frag_acc = torch.mean((torch.argmax(pred_frags, dim=1) == frag_labels).float())
        frag_baseline = 1 - torch.sum(frag_labels)/len(frag_labels)

        self.log(
            "Train CELoss",
            loss.detach()
        )
        self.log(
            "Train Frag Loss",
            frag_loss.detach()
        )
        self.log(
            "Train AA Acc.",
            aa_acc.detach()
        )
        self.log(
            "Train Pep. Acc.",
            pep_acc.detach()
        )
        self.log(
            "Train Frag Acc.",
            frag_acc.detach()
        )

        return loss + frag_loss

    def validation_step(self, batch, b_idx, d_idx):
        torch.set_grad_enabled(True)

        spectra, precursors, sequences, frag_labels = batch
        preds, pred_prec, pred_frags = self._forward_step(spectra, precursors, sequences)
        preds_seqs = torch.argmax(preds, dim=2)[:,:-1]
        preds = preds[:,:-1,:].reshape(-1, len(self.tokenizer) + 1)
        true_seqs = batch[2][:,:]
        loss = self.CELoss(preds,true_seqs.flatten())

        true_prec = batch[1][:,0]/1000
        mse_loss = self.MSELoss(pred_prec, true_prec)

        pred_frags = pred_frags[:,1:,:].reshape(-1, 2)
        frag_loss = self.fragCELoss(pred_frags, frag_labels)
        frag_acc = torch.mean((torch.argmax(pred_frags, dim=1) == frag_labels).float())
        frag_baseline = 1 - torch.sum(frag_labels)/len(frag_labels)

        correct_aas = torch.logical_or(preds_seqs == true_seqs, true_seqs == 0)
        padding = torch.sum(true_seqs == 0)
        aa_acc = (torch.sum(correct_aas.float()) - padding) / torch.sum(true_seqs != 0)

        correct_peps = torch.mean(correct_aas.float(), dim=1) == 1
        pep_acc = torch.mean(correct_peps.float())

        self.log(
            f"Val CELoss",
            loss.detach(),
            on_epoch=True,
            sync_dist=True
        )
        self.log(
            f"Val Frag Loss",
            frag_loss.detach(),
            on_epoch=True,
            sync_dist=True
        )
        self.log(
            f"Val AA Acc.",
            aa_acc.detach(),
            on_epoch=True,
            sync_dist=True
        )
        self.log(
            f"Val Pep. Acc.",
            pep_acc.detach(),
            on_epoch=True,
            sync_dist=True
        )
        self.log(
            f"Val Frag Acc.",
            frag_acc.detach(),
            on_epoch=True,
            sync_dist=True
        )

        return loss, pep_acc + frag_loss

    def predict_step(self, batch, *args):
        torch.set_grad_enabled(True)

        spectra, precursors, sequences, frag_labels = batch

        cur_sequences = torch.empty((len(sequences),0), dtype=torch.int32, device=torch.device('cuda'))
        aa_conf = torch.empty((len(sequences),0), dtype=torch.int32, device=torch.device('cuda'))
        for i in range(len(sequences[0]) + 1):
            preds, _, _ = self._forward_step(spectra, precursors, cur_sequences)
            next_aa_scores = torch.softmax(preds[:,-1,:], 1)
            next_aas = torch.argmax(next_aa_scores, 1)
            aa_conf = torch.cat([aa_conf, torch.reshape(next_aa_scores[range(len(next_aas)),next_aas], (-1, 1))], dim=1)
            cur_sequences = torch.cat([cur_sequences, torch.reshape(next_aas, (-1, 1))], dim=1)

        preds_seqs = cur_sequences
        true_seqs = batch[2]

        pred_seqs = self.tokenizer.detokenize(preds_seqs)
        true_seqs = self.tokenizer.detokenize(true_seqs)

        pep_conf = []
        trimmed_pred_seqs = []
        for pred_pep, pep_aa_conf in zip(pred_seqs, aa_conf):
            trimmed_pep = pred_pep.split('$')[0]
            trimmed_pred_seqs.append(trimmed_pep)
            pep_conf.append(torch.mean(pep_aa_conf[:len(trimmed_pep) + 1]))

        return trimmed_pred_seqs, true_seqs, pep_conf, aa_conf

    def configure_optimizers(self):
        """
        Initialize the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.lr_decay)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=10000, max_iters=100000
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor
