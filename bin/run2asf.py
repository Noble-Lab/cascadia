from pyteomics import mzml, auxiliary, mass
import matplotlib.pyplot as plt
import numpy as np
from pyteomics import mzml, auxiliary, mass

seq_to_val = {}
raw = '20230406_OLEP08_MMCC_1ug_MB_24min_AS_10ms_4Th_I_1.mzML'
seq_to_val['PEPTIDEK'] = False
f_to_mzrt_to_pep = {}
count = 0
part = 0
with mzml.read(raw) as reader:
    for spec in reader:
        if spec['ms level'] == 2:
            window = spec['precursorList']['precursor'][0]['isolationWindow']
            window_center = window['isolation window target m/z']
            lower_offset = window['isolation window lower offset']
            upper_offset = window['isolation window upper offset']
            cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
            if count > 1200000:
                break
            elif count % 50000 == 0:
                part += 1
                print(part)
                f_to_mzrt_to_pep[raw+'_'+str(part)] = {}
            elif count % 1000 == 0:
                print(count)
            count += 1
            key = (int(window_center/10), int(cur_rt/10))
            if key in f_to_mzrt_to_pep[raw+'_'+str(part)]:
                f_to_mzrt_to_pep[first_fname+'_'+str(part)][key].append(('PEPTIDEK', window_center, cur_rt, 1))
            else:
                f_to_mzrt_to_pep[first_fname+'_'+str(part)][key] = [('PEPTIDEK', window_center, cur_rt, 1)]

print(count)

max_widths = [1, 3, 5]
rt_width = 2.0
max_peaks = 150

count = 0
found, found_25, found_35, filtered = 0, 0, 0, 0
train = 0
val = 0
all_peps = []
top_peps = []
a_found = []
a_tics = []
for fname in f_to_mzrt_to_pep.keys():
  prec_to_spec = {}
  print(fname)
  count = 0
  d_tics = []
  d_found_ions = []
  scores = []
  centers_found = 0
  with mzml.read('../data/astral/hela/' + fname[:-2]+'.mzML') as reader:
    for spec in reader:
      if centers_found % 1000 == 0:
        print(centers_found, len(prec_to_spec))
      if spec['ms level'] == 1:
        cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
        for scan_rt in [int(cur_rt/10) - 1, int(cur_rt/10), int(cur_rt/10) + 1]:
          for scan_window in range(30,1100):
            if (scan_window, scan_rt) in f_to_mzrt_to_pep[fname]:
              for seq, mz, rt, charge in f_to_mzrt_to_pep[fname][(scan_window, scan_rt)]:
                if np.abs(rt - cur_rt) < rt_width:
                  prec = mz
                  mzs = spec['m/z array']
                  intensities = spec['intensity array']

                  sorted_intensity_idxs = np.argsort(intensities)[-max_peaks:]
                  intensities = intensities[sorted_intensity_idxs]
                  mzs = mzs[sorted_intensity_idxs]

                  sorted_mz_idxs = np.argsort(mzs)
                  intensities = intensities[sorted_mz_idxs]
                  mzs = mzs[sorted_mz_idxs]

                  if (fname, mz, rt, charge) not in prec_to_spec:
                    prec_to_spec[(fname, mz, rt, charge)] = {}
                  if 'ms1_scans' not in prec_to_spec[(fname, mz, rt, charge)]:
                    prec_to_spec[(fname, mz, rt, charge)]['ms1_scans'] = []
                    prec_to_spec[(fname, mz, rt, charge)]['ms1_rts'] = []
                  prec_to_spec[(fname, mz, rt, charge)]['ms1_scans'].append([x for x in zip(mzs, intensities)])
                  prec_to_spec[(fname, mz, rt, charge)]['ms1_rts'].append(cur_rt - rt)

      else:
        window = spec['precursorList']['precursor'][0]['isolationWindow']
        window_center = window['isolation window target m/z']
        lower_offset = window['isolation window lower offset']
        upper_offset = window['isolation window upper offset']
        cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']

        for scan_rt in [int(cur_rt/10) - 1, int(cur_rt/10), int(cur_rt/10) + 1]:
          for scan_window in range(int((window_center - lower_offset)/10) - 1, int((window_center + upper_offset)/10) + 1):
            if (scan_window, scan_rt) in f_to_mzrt_to_pep[fname]:
              for seq, mz, rt, charge in f_to_mzrt_to_pep[fname][(scan_window, scan_rt)]:
                if mz > window_center - lower_offset and mz < window_center + upper_offset and np.abs(rt - cur_rt) < .001:
                  if 'U' in seq:
                    print('Uh oh - U')

                  centers_found += 1

                  prec = mz
                  mzs = spec['m/z array']
                  intensities = spec['intensity array']

                  sorted_intensity_idxs = np.argsort(intensities)[-max_peaks:]
                  intensities = intensities[sorted_intensity_idxs]
                  mzs = mzs[sorted_intensity_idxs]

                  sorted_mz_idxs = np.argsort(mzs)
                  intensities = intensities[sorted_mz_idxs]
                  mzs = mzs[sorted_mz_idxs]

                  ions = [1]
                  seq = ''.join(filter(str.isalpha, seq))
                  all_peps.append(seq)

                  tic = np.sum(intensities)

                  d_tics.append(fic/tic)
                  d_found_ions.append(n_found/len(ions))

                  if (fname, mz, rt, charge) not in prec_to_spec:
                    prec_to_spec[(fname, mz, rt, charge)] = {}
                  prec_to_spec[(fname, mz, rt, charge)]['prec'] = prec
                  prec_to_spec[(fname, mz, rt, charge)]['seq'] = seq
                  prec_to_spec[(fname, mz, rt, charge)]['val'] = seq_to_val[seq]
                  if 'scans' not in prec_to_spec[(fname, mz, rt, charge)]:
                    prec_to_spec[(fname, mz, rt, charge)]['scans'] = []
                    prec_to_spec[(fname, mz, rt, charge)]['rts'] = []
                  prec_to_spec[(fname, mz, rt, charge)]['scans'].append([x for x in zip(mzs, intensities)])
                  prec_to_spec[(fname, mz, rt, charge)]['rts'].append(cur_rt - rt)

            elif mz > window_center - lower_offset and mz < window_center + upper_offset and np.abs(rt - cur_rt) < rt_width:

              mzs = spec['m/z array']
              intensities = spec['intensity array']

              sorted_intensity_idxs = np.argsort(intensities)[-150:]
              intensities = intensities[sorted_intensity_idxs]
              mzs = mzs[sorted_intensity_idxs]

              sorted_mz_idxs = np.argsort(mzs)
              intensities = intensities[sorted_mz_idxs]
              mzs = mzs[sorted_mz_idxs]

              if (fname, mz, rt, charge) not in prec_to_spec:
                prec_to_spec[(fname, mz, rt, charge)] = {}
              if 'scans' not in prec_to_spec[(fname, mz, rt, charge)]:
                prec_to_spec[(fname, mz, rt, charge)]['scans'] = []
                prec_to_spec[(fname, mz, rt, charge)]['rts'] = []
              prec_to_spec[(fname, mz, rt, charge)]['scans'].append([x for x in zip(mzs, intensities)])
              prec_to_spec[(fname, mz, rt, charge)]['rts'].append(cur_rt - rt)

  for ms_augment_level in ['ms1']:
    for max_width in max_widths:
      out = f"astral_dia_w{max_width}_{ms_augment_level}_hela_A.asf"
      out = open(out, 'a')

      for key, value in prec_to_spec.items():

        count += 1

        fname, mz, rt, charge = key
        prec = value['prec']
        seq = value['seq']
        scans = np.array(value['scans'], dtype=object)
        rts = np.array(value['rts'])
        ms1_scans = np.array(value['ms1_scans'], dtype=object)
        ms1_rts = np.array(value['ms1_rts'])
        val = value['val']

        abs_rts = [np.abs(x) for x in rts]
        sorted_rt_idxs = np.argsort(abs_rts)[:max_width]
        rts = rts[sorted_rt_idxs]
        scans = scans[sorted_rt_idxs]

        abs_ms1_rts = [np.abs(x) for x in ms1_rts]
        sorted_ms1_rt_idxs = np.argsort(abs_ms1_rts)[:max_width]
        ms1_rts = ms1_rts[sorted_ms1_rt_idxs]
        ms1_scans = ms1_scans[sorted_ms1_rt_idxs]

        a_found.append(fragments)
        a_tics.append(ion_current)


        for charge in ['2', '3', '4']:

          if fragments > -1:

            found += 1

            out.write("BEGIN IONS\n")
            out.write(f"TITLE={str(count)}+{str(ion_current)}_{str(fragments)}\n")
            out.write(f"PEPMASS={prec}\n")
            out.write(f"CHARGE={charge}\n")
            out.write(f"SCAN={count}\n")
            out.write(f"SEQ={seq.replace('C', 'C+57.021')}\n")

            for scan, cur_rt in zip(scans,rts):
              for mz, intensity in scan:
                out.write(f"{mz}\t{intensity}\t{cur_rt}\t2\n")

            if ms_augment_level == 'ms1':
              for scan, cur_rt in zip(ms1_scans, ms1_rts):
                for mz, intensity in scan:
                  if np.abs(mz - prec) < 10:
                    out.write(f"{mz}\t{intensity}\t{cur_rt}\t1\n")

            out.write("END IONS\n")

            if found % 1000 == 0:
              print(found, found_25, found_35, filtered)

          else:
            filtered += 1

      out.close()

print(count, found, train, val)
