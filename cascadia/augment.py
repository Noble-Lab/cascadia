from pyteomics import mzml, auxiliary, mass
import numpy as np
import os

def get_centers(mzml_file):
    f_to_mzrt_to_pep = {}
    max_mz = 0
    count = 0
    part = 0
    with mzml.read(mzml_file) as reader:
        for spec in reader:
            if spec['ms level'] == 2:
                window = spec['precursorList']['precursor'][0]['isolationWindow']
                window_center = window['isolation window target m/z']
                lower_offset = window['isolation window lower offset']
                upper_offset = window['isolation window upper offset']
                cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
                if count % 50000 == 0:
                    part += 1
                    f_to_mzrt_to_pep[part] = {}
                count += 1
                key = (int(window_center/10), int(cur_rt/10))
                max_mz = max(max_mz,int(window_center/10))
                if key in f_to_mzrt_to_pep[part]:
                    f_to_mzrt_to_pep[part][key].append((window_center, cur_rt, 1))
                else:
                    f_to_mzrt_to_pep[part][key] = [(window_center, cur_rt, 1)]
    return f_to_mzrt_to_pep, max_mz

def extract_spectra(mzml_file, f_to_mzrt_to_pep, part, top_n, time_width, max_mz):
    prec_to_spec = {}
    with mzml.read(mzml_file) as reader:
        for spec in reader:
            cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
            if spec['ms level'] == 1:
                for scan_rt in range(int(cur_rt/10) - 1, int(cur_rt/10) + 1):
                    for scan_window in range(max_mz+1):
                        if (scan_window, scan_rt) in f_to_mzrt_to_pep[part]:
                            for mz, rt, charge in f_to_mzrt_to_pep[part][(scan_window, scan_rt)]:
                                if np.abs(rt - cur_rt) < time_width: 
                                    mzs = spec['m/z array']
                                    intensities = spec['intensity array']

                                    sorted_intensity_idxs = np.argsort(intensities)[-top_n:]
                                    intensities = intensities[sorted_intensity_idxs]
                                    mzs = mzs[sorted_intensity_idxs]

                                    sorted_mz_idxs = np.argsort(mzs)
                                    intensities = intensities[sorted_mz_idxs]
                                    mzs = mzs[sorted_mz_idxs]

                                    if (mz, rt, charge) not in prec_to_spec:
                                        prec_to_spec[(mz, rt, charge)] = {}
                                    if 'ms1_scans' not in prec_to_spec[(mz, rt, charge)]:
                                        prec_to_spec[(mz, rt, charge)]['ms1_scans'] = []
                                        prec_to_spec[(mz, rt, charge)]['ms1_rts'] = []
                                    prec_to_spec[(mz, rt, charge)]['ms1_scans'].append([x for x in zip(mzs, intensities)])
                                    prec_to_spec[(mz, rt, charge)]['ms1_rts'].append(cur_rt - rt)
            elif spec['ms level'] == 2:
                window = spec['precursorList']['precursor'][0]['isolationWindow']
                window_center = window['isolation window target m/z']
                lower_offset = window['isolation window lower offset']
                upper_offset = window['isolation window upper offset']
            
                for scan_rt in range(int(cur_rt/10) - 1, int(cur_rt/10) + 1):
                    for scan_window in range(int((window_center - lower_offset)/10) - 1, int((window_center + upper_offset)/10) + 1):
                        if (scan_window, scan_rt) in f_to_mzrt_to_pep[part]:
                            for mz, rt, charge in f_to_mzrt_to_pep[part][(scan_window, scan_rt)]:
                                in_mz = mz > window_center - lower_offset and mz < window_center + upper_offset
                                rt_diff = np.abs(rt - cur_rt)
                                if in_mz and rt_diff < time_width:
                                    mzs = spec['m/z array']
                                    intensities = spec['intensity array']

                                    sorted_intensity_idxs = np.argsort(intensities)[-top_n:]
                                    intensities = intensities[sorted_intensity_idxs]
                                    mzs = mzs[sorted_intensity_idxs]

                                    sorted_mz_idxs = np.argsort(mzs)
                                    intensities = intensities[sorted_mz_idxs]
                                    mzs = mzs[sorted_mz_idxs]
                      
                                    if (mz, rt, charge) not in prec_to_spec:
                                        prec_to_spec[(mz, rt, charge)] = {}
                                    if 'scans' not in prec_to_spec[(mz, rt, charge)]:
                                        prec_to_spec[(mz, rt, charge)]['scans'] = []
                                        prec_to_spec[(mz, rt, charge)]['rts'] = []
                                        prec_to_spec[(mz, rt, charge)]['window_width'] = max(lower_offset, upper_offset) 
                                    prec_to_spec[(mz, rt, charge)]['scans'].append([x for x in zip(mzs, intensities)])
                                    prec_to_spec[(mz, rt, charge)]['rts'].append(cur_rt - rt)
    return prec_to_spec

def write_asf(outfile, prec_to_spec, scan_width, max_pep_length, max_charge):
    out = open(outfile, 'a')
    skipped = 0
    for key, value in prec_to_spec.items():
        if 'ms1_scans' not in value:
            skipped += 1
            continue
        prec, rt, charge = key
        scans = np.array(value['scans'], dtype=object)
        rts = np.array(value['rts'])
        ms1_scans = np.array(value['ms1_scans'], dtype=object)
        ms1_rts = np.array(value['ms1_rts'])
        window_width = value['window_width']

        abs_rts = [np.abs(x) for x in rts]
        sorted_rt_idxs = np.argsort(abs_rts)[:scan_width]
        rts = rts[sorted_rt_idxs]
        scans = scans[sorted_rt_idxs]

        abs_ms1_rts = [np.abs(x) for x in ms1_rts]
        sorted_ms1_rt_idxs = np.argsort(abs_ms1_rts)[:scan_width]
        ms1_rts = ms1_rts[sorted_ms1_rt_idxs]
        ms1_scans = ms1_scans[sorted_ms1_rt_idxs]
        
        count = 0
        for charge in range(1,max_charge+1):
            count += 1

            out.write("BEGIN IONS\n")
            out.write(f"TITLE={str(count)}\n")
            out.write(f"PEPMASS={prec}\n")
            out.write(f"CHARGE={charge}\n")
            out.write(f"SCAN={count}\n")
            out.write(f"RT={rt}\n")
            out.write(f"SEQ={'K'*max_pep_length}\n")
            
            for scan, cur_rt in zip(scans,rts):
                for mz, intensity in scan:
                    out.write(f"{mz}\t{intensity}\t{cur_rt}\t2\n")
            
            for scan, cur_rt in zip(ms1_scans, ms1_rts):
                for mz, intensity in scan:
                    if np.abs(mz - prec) < window_width + 1:
                        out.write(f"{mz}\t{intensity}\t{cur_rt}\t1\n")

            out.write("END IONS\n")   
    out.close()

def augment_spectra(mzml_file, top_n = 150, scan_width = 1, time_width = 3, max_pep_length = 20, max_charge = 3):
    outfile = "temp.asf"
    if os.path.exists(outfile):
        os.remove(outfile)
    f_to_mzrt_to_pep, max_mz = get_centers(mzml_file)
    for part in f_to_mzrt_to_pep.keys():
        prec_to_spec = extract_spectra(mzml_file, f_to_mzrt_to_pep, part, top_n, time_width, max_mz)
        print(f'Generating {len(prec_to_spec) * max_charge} augmented spectra')
        write_asf(outfile, prec_to_spec, scan_width, max_pep_length, max_charge)
    return outfile