# File Formats

### Spectrum Input 

For inference, Cascadia takes as input spectra in [mzML](https://www.sciencedirect.com/science/article/pii/S1535947620313876?via%3Dihub) format, which is a standardized and commonly used machine readable XML file format developed by the Proteomics Standards Initiative (PSI). 

Raw vendor files can be converted to mzML format using the [msConvert](https://proteowizard.sourceforge.io/tools/msconvert.html) tool in [ProteoWizard](https://proteowizard.sourceforge.io/index.html). We suggest using the adding the peakPicking filter at both the MS1 and MS2 levels using the vendor algorithm, and otherwise using default parameters. 

### Output format 

Cascadia produces as output a generic tab-delimited text file format refered to as ssl (spectrum sequence list). A full description of this file format is available
[here](https://skyline.ms/wiki/home/software/BiblioSpec/page.view?name=BiblioSpec%20input%20and%20output%20file%20formats). The file contains one row for each identification. The columns contain the following information for each prediction: 

| | |
| ---- | ---- |
| __file__ | The mzML file containing the identified spectrum |
| __scan__ | The ScanID for the center of the detection |
| __charge__ | Charge state of the precursor |
| __sequence__ | Peptide sequence |
| __score__ | Cascadia confidence score assigned to this prediction |
| __retention-time__ | The retention time (in minutes) when this peptide was detected |


To visualize the results from Cascadia sequencing, this .ssl file can be coverted into a spectral library and loaded into [skyline](https://skyline.ms/wiki/home/software/BiblioSpec/page.view?name=default). 

### .asf format 
To train Cascadia, the user needs to supply a collection of labeled augmented spetra. Augmented spectra are represented in `.asf` format. This format is a simple extension of the [Mascot Generic Format (MGF)](https://www.matrixscience.com/help/data_file_help.html) format for spectra with annotations. Each augmented spectrum starts with a set of annotations, which for Cascadia should at a minimum contain the fields '`SEQ`, `CHARGE`, and `PEPMASS`. `PEPMASS` should be the center m/z for the isolation window, while `SEQ` and `CHARGE` should contain the sequence and charge state for the precursor idenified in the augmented spectrum. 

Following the anotations is a list of tab-delimited 4-tuples, with one row for each peak in the augmented spectrum. The augmented spectrum should contain peaks from the 2w+1 MS2 scans surrounding the central scan where the peptide is identified (where w is the augmentation width), along with the peaks from the corresponding MS1 scans that fall within the isolation window. Each row contains the m/z and intensity for a single peak, the same as in an MGF file, along with the relative position of the scan this peak is from compared to the central scan in the augmented spectrum (an integer in the range [-w, w]), and either a 1 or a 2 representing whether it is an MS1 or MS2 peak.

An example augmented spectrum with w=1 is shown below:

```
BEGIN IONS
TITLE=1
PEPMASS=402.5
CHARGE=1
SCAN=1
RT=0.31070598402
SEQ=PEPTIDEK
185.07986450195312      0.666241404428991       0     2
367.8453063964844       0.4087404090593467      0     2
400.98358154296875      0.8838386047632094      0     2
401.1849060058594       0.4504917813994659      0     2
401.2586669921875       0.42375381912879023     0     2
401.9834289550781       0.6479885582525154      0     2
418.99261474609375      0.6245163448036832      0     2
606.7568359378328       0.41198908195979644     0     2
606.756835937573        0.41198908195979644     0     2
843.2614135742188       0.41591027114618606     0     2
946.1708374023438       0.430206516710194       0     2
185.07986450195312      0.666241404428991       1     2
367.8453063964844       0.4087404090593467      1     2
400.98358154296875      0.8838386047632094      1     2
401.1849060058594       0.4504917813994659      1     2
401.2586669921875       0.42375381912879023     1     2
401.9834289550781       0.6479885582525154      1     2
418.99261474609375      0.6245163448036832      1     2
185.07986450195312      0.666241404428991       -1    2
367.8453063964844       0.4087404090593467      -1    2
400.98358154296875      0.8838386047632094      -1    2
401.1849060058594       0.4504917813994659      -1    2
401.2586669921875       0.42375381912879023     -1    2
401.9834289550781       0.6479885582525154      -1    2
418.99261474609375      0.6245163448036832      -1    2
606.7568359375737       0.41198908195979644     -1    2
843.2614135742188       0.41591027114618606     -1    2
946.1708374023438       0.430206516710194       -1    2
400.1812744140625       0.2918983249625358      0     1
400.3501892089844       0.3196363715025925      0     1
401.2091979980469       0.34867104752184375     0     1
403.2320556640625       0.30672175589489503     0     1
402.2713624687576       0.2951177103313615      -1    1
403.17193603515625      0.362321942870444       -1    1
END IONS
```