# -*- encoding: utf-8 -*-
'''
@File    :   0_statistic_cancer_geneNums.py
@Time    :   2019/12/13 10:33:53
@Author  :   yetaoyu 
@Version :   1.0
@Contact :   taoyuye@gmail.com
@Desc    :   Statistics the number of genes and save the gene name of each cancer to the directory "/geneNameCancer/" 
'''

# here put the import lib
import numpy as np 
import pandas as pd
import sys
import os
import glob
import datetime
import cv2

# Extract the gene names associated with each cancer dataset and return the screened gene set
def read_gene_names(path, cancer_name):
	os.chdir(path)
	genes = list()
	chrom_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r', encoding='ISO-8859-1') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []

	# Read each sample file to get the gene name: current_gene_list
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name

	for i in range(0, len(A_sample_id_list)): # Iterate over each sample
		sample_id = A_sample_id_list[i]
		sample_file = A_sample_name_list[i]	
		# The specific path of each samples
		sample_path = path + "/" + sample_file

		if os.path.exists(sample_path):
			begin = datetime.datetime.now()
			current_gene_list = list()
			with open(sample_path, 'r', encoding='ISO-8859-1') as f:
				title_line = f.readline()
				title_cols = title_line.split('\t')
				hugo_symbol_index = title_cols.index('Hugo_Symbol')
				variant_type_index = title_cols.index('Variant_Type')
				chromosome_index = title_cols.index('Chromosome')

				for line in f:
					line = line.strip()
					cols = line.split()

					gene = cols[hugo_symbol_index]
					variant = cols[variant_type_index]
					chrom = cols[chromosome_index]
					if (chrom in chrom_list) and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

	return genes

def main():
	data_path = os.getcwd()

	gene_name_path = data_path + '/geneNameCancer/' 
	if not os.path.exists(gene_name_path):
		os.makedirs(gene_name_path)

	cancer_directories= ['UCS', 'KICH', 'UCEC', 'OV', 'GBM', 'LGG', 'SKCM', 'LAML', 'KIRC', 'HNSC', 'TGCT', 'THCA', 'PCPG', 'COAD', 'THYM', 'READ', 'CHOL', 'BLCA', 'PRAD', 'LUAD', 'STAD', 'CESC', 'UVM', 'GBMLGG', 'BRCA', 'LUSC', 'SARC', 'LIHC', 'ACC', 'ESCA', 'STES', 'PAAD', 'KIPAN', 'KIRP', 'COADREAD', 'DLBC']
	# cancer_directories = [name for name in os.listdir(".") if os.path.isdir(name)] # Cancer directory name
	print("cancer_directories=",cancer_directories)

	for cancer in range(len(cancer_directories)):
		
		cancer_name = cancer_directories[cancer]
		# print("cancer_name",cancer_name)

		fopen = open(gene_name_path + cancer_name + "_gene_name.txt", 'w')
		
		cancer_directory_path = os.path.join(data_path, cancer_name)
		cancer_genes = read_gene_names(cancer_directory_path, cancer_name) # gene set for each cancer

		for gene in cancer_genes:
			fopen.write(str(gene) + '\n')

		os.chdir(data_path)
		print(" The cancer_name ", cancer_name, "concluding ", len(cancer_genes), "genes")

		fopen.close()

if __name__ == "__main__":
	main()
