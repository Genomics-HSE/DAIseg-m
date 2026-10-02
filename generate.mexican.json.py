#!/usr/bin/env python3

import json
import os
import sys

PED_DIR = "/home/share/human.data/1000GP/pedigree.grch38"

def read_sample_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_base_config(chrom, outdir, yri, pel, ibs, mxl):
    return {
        "description": "DAIseg.mexicans configuration to run",
        "CHROM": f"chr{chrom}",
        "output": f"MXL.grch38.chr{chrom}",
        "prefix": f"{outdir}",
        "files": {
            "neand_files": {
                "Vindija33.19": {
                    "bed": f"/home/share/human.data/neand/33.19.grch38/bed/chr{chrom}_mask.bed.gz",
                    "vcf": f"/home/share/human.data/neand/33.19.grch38/chr{chrom}.vcf.gz"
                },
                "Altai": {
                    "bed": f"/home/share/human.data/neand/altai.grch38/bed/chr{chrom}_mask.bed.gz",
                    "vcf": f"/home/share/human.data/neand/altai.grch38/chr{chrom}.vcf.gz"
                },

                "Chagyrskaya-Phalanx": { 
                    "bed": f"/home/share/human.data/neand/Chagyrskaya.grch38/bed/chr{chrom}_mask.bed.gz", 
                    "vcf": f"/home/share/human.data/neand/Chagyrskaya.grch38/chr{chrom}.vcf.gz" 
                }
            },
            "1000GP_files": {
                "bed": f"/home/share/human.data/1000GP/1000GP.grch38/bed/chr{chrom}.clean.bed",
                "vcf": f"1kG_filtered.chr{chrom}.grch38.bcf",
                "vcf_initial": f"/home/share/human.data/1000GP/1000GP.grch38/CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
            },
            "ancestral": {
                "fasta": f"/home/share/human.data/Anc.fa/homo_sapiens_ancestor_GRCh38/homo_sapiens_ancestor_{chrom}.fa"
            },
            "reference": {
                "fasta": "/home/share/human.data/ref.fa/grch38.fasta/GRCh38_full_analysis_set_plus_decoy_hla.fa"
            },
            "chr_lengths": "/home/share/human.data/ref.fa/grch38.lengths/hg38.chrom.sizes"
        },
        "samples": {
            "Africans": yri,
            "Americans": pel,
            "Europeans": ibs,
            "Mexicans": mxl,
            "neand": [
                "Vindija33.19",
                "Altai",
                "Chagyrskaya-Phalanx"
            ]
        },
        "parameters_initial": {
            "admixture_nd": 0.02,
            "admixture_modern": [0.5, 0.4, 0.1],
            "introgression_time": 55000,
            "rr": 1e-08,
            "mutation": 1.25e-08,
            "window_length": 1000,
            "generation_time": 29,
            "t_n_c": 550000,
            "t_af_c": 70000,
            "t_introgression_c": 55000,
            "t_ea_c": 41000,
            "t_mexicans_c": 500,
            "t_introgression": 55000,
            "t_mexicans": 500
        },
        "window_callability": {
            "Thousand_genomes": f"coverage_1kG.chr{chrom}.grch38.bed",
            "Nd_1k_genomes": f"coverage_1kG.nd.chr{chrom}.grch38.bed"
        },
        "data": f"prep.chr{chrom}.grch38.tsv",
        "gaps": "/home/share/human.data/ref.fa/gaps.grch38/gap.txt"
    }

def main():
    outdir = sys.argv[1]

    yri = read_sample_list(f"{PED_DIR}/YRI.unrelated.2504.txt")
    pel = read_sample_list(f"{PED_DIR}/PEL.unrelated.2504.txt")
    ibs = read_sample_list(f"{PED_DIR}/IBS.unrelated.hc3202.txt")
    mxl = read_sample_list(f"{PED_DIR}/MXL.unrelated.hc3202.txt")

    os.makedirs(outdir, exist_ok=True)

    for chrom in range(1, 23):
        config = get_base_config(chrom, outdir, yri, pel, ibs, mxl)
        out_file = os.path.join(outdir, f"MXL.grch38.chr{chrom}.json")
        with open(out_file, "w") as f:
            json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
