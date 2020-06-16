# -*- encoding: utf-8 -*-
'''
@File    :   2_statistic_same_gene_chrom_noselect.py
@Time    :   2019/12/13 10:38:34
@Author  :   yetaoyu 
@Version :   1.0
@Contact :   taoyuye@gmail.com
@Desc    :   Statistics the gene name list of 36 types of cancer under the same chromosome 
	and save it to the corresponding chromosome file under "ChromosomeGene/All/", 
	such as "ChromosomeGene/All/chrom01.txt"
'''

# here put the import lib
import numpy as np 
import pandas as pd
import sys
import os
import glob
import datetime
import cv2

def read_gene_names(path, chrom_file_name):
	os.chdir(path)
	chrom_genes = list()
	with open(chrom_file_name, 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			chrom_genes.append(cols[0])
	return chrom_genes

def main():
	current_path = os.getcwd()

	chrom_dict = {0: 'chrom01',1: 'chrom02',2: 'chrom03',3: 'chrom04',4: 'chrom05',5: 'chrom06',6: 'chrom07',7: 'chrom08',8: 'chrom09',9: 'chrom10',10: 'chrom11',11: 'chrom12',12: 'chrom13',13: 'chrom14',14: 'chrom15',15: 'chrom16',16: 'chrom17',17: 'chrom18',18: 'chrom19',19: 'chrom20',20: 'chrom21',21: 'chrom22',22: 'chromX',23: 'chromY'}

	data_path = os.path.join(current_path, "ChromosomeGene") #  ./ChromosomeGene
	print("data_path=",data_path)
	os.chdir(data_path) 
	cancer_directories = [name for name in os.listdir(".") if os.path.isdir(name)]
	print("cancer_directories=",cancer_directories)

	chrom_path = os.path.join(data_path, "ACC")
	os.chdir(chrom_path) 
	chrom_directories = [name for name in os.listdir(".") if os.path.isfile(name)] # chrom01.txt
	chrom_directories.sort()
	print("chrom_directories=",chrom_directories)

	os.chdir(current_path)

	for chrom in range(len(chrom_directories)):
		genes = []
		for cancer in range(len(cancer_directories)):
			cancer_directory_path = os.path.join(data_path, cancer_directories[cancer])

			#print("cancer_directory_path=", cancer_directory_path)
			#print("chrom_directories[",chrom,"]")
			chrom_genes = read_gene_names(cancer_directory_path, chrom_directories[chrom]) 
			genes.extend(set(chrom_genes) - set(genes))
			os.chdir(data_path)
		
		genes = list(set(genes))
		print("chrom",chrom,"concluding",len(genes),"genes!")

		# create a new file: eg. chrom01.txt
		total_path = data_path + "/All/"
		if not os.path.exists(total_path):
			os.makedirs(total_path)
		fopen = open(total_path + chrom_dict[chrom] + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	os.chdir(current_path)


if __name__ == "__main__":
	main()
