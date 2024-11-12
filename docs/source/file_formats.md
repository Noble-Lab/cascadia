# File Formats

### Spectrum Input 



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


