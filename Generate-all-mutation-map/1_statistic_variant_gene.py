#!/usr/bin/env Python
# coding=utf-8
'''
@File    :   1_statistic_variant_gene.py
@Time    :   2019/12/13 10:33:03
@Author  :   yetaoyu 
@Version :   1.0
@Contact :   taoyuye@gmail.com
@Desc    :   Statistics the gene name list of each chromosome for each cancer separately 
	and save it to the corresponding cancer directory under "ChromosomeGene/", 
	such as "ChromosomeGene/ACC/Chrom01.txt"
''' 

# here put the import lib
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

import numpy as np 
import pandas as pd
import os
import glob
import datetime
import cv2

def read_gene_names1(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "1":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)

	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom01" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names2(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "2":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom02" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names3(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "3":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom03" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names4(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "4":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom04" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names5(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "5":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom05" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names6(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "6":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom06" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names7(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "7":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom07" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names8(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "8":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom08" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names9(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "9":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom09" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names10(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "10":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom10" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names11(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "11":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom11" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names12(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "12":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom12" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names13(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "13":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom13" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names14(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "14":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom14" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names15(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "15":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom15" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names16(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "16":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom16" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names17(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "17":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom17" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names18(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "18":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom18" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names19(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "19":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom19" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names20(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "20":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom20" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names21(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "21":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom21" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_names22(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "22":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chrom22" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_namesX(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "X":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chromX" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def read_gene_namesY(path, cancer_name):
	os.chdir(path)
	genes = list()

	# sample names
	file_name_list = list()
	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			file_name_list.append(cols[1])

	A_sample_id_list = []
	A_sample_name_list = []	
	fig_path = "../ChromosomeGene/"

	# Read each sample file to get the gene names
	for i in range(len(file_name_list)):
		A_sample_id_list.append(file_name_list[i].split('.maf')[0]) # Sample id
		A_sample_name_list.append(file_name_list[i]) # Sample name
		if not os.path.exists(fig_path+cancer_name):
			os.makedirs(fig_path+cancer_name)
	
	for i in range(0, len(A_sample_id_list)): 
		#print("now sample id", A_sample_id_list[i])

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

					if chrom == "Y":
						if(variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
							current_gene_list.append(gene)	
	
		genes.extend(set(current_gene_list) - set(genes))	# only including the genes that are not previously found

		fopen = open(fig_path + cancer_name + "/chromY" + ".txt", 'w')
		for gene in range(len(genes)):
			fopen.write(genes[gene] + '\n')
		fopen.close()

	return genes

def main():
	current_path = os.getcwd()		
	
	#cancer_directories = [name for name in os.listdir(".") if os.path.isdir(name)] # cancer directories
	#print("cancer_directories=",cancer_directories)

	cancer_directories= ['UCS', 'KICH', 'UCEC', 'OV', 'GBM', 'LGG', 'SKCM', 'LAML', 'KIRC', 'HNSC', 'TGCT', 'THCA', 'PCPG', 'COAD', 'THYM', 'READ', 'CHOL', 'BLCA', 'PRAD', 'LUAD', 'STAD', 'CESC', 'UVM', 'GBMLGG', 'BRCA', 'LUSC', 'SARC', 'LIHC', 'ACC', 'ESCA', 'STES', 'PAAD', 'KIPAN', 'KIRP', 'COADREAD', 'DLBC']
	
	os.chdir(current_path)

	for cancer in range(len(cancer_directories)):
		cancer_name = cancer_directories[cancer]
		print("cancer_name",cancer_name)
		
		cancer_directory_path = os.path.join(current_path, cancer_name)
		cancer_gene1 = read_gene_names1(cancer_directory_path, cancer_name) # Get gene sets for each cancer
		os.chdir(current_path) 
		print(" The chrom1 of cancer_name ", cancer_name, "concluding ", len(cancer_gene1), "genes")

		cancer_gene2 = read_gene_names2(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom2 of cancer_name ", cancer_name, "concluding ", len(cancer_gene2), "genes")

		cancer_gene3 = read_gene_names3(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom3 of cancer_name ", cancer_name, "concluding ", len(cancer_gene3), "genes")

		cancer_gene4 = read_gene_names4(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom4 of cancer_name ", cancer_name, "concluding ", len(cancer_gene4), "genes")

		cancer_gene5 = read_gene_names5(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom5 of cancer_name ", cancer_name, "concluding ", len(cancer_gene5), "genes")

		cancer_gene6 = read_gene_names6(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom6 of cancer_name ", cancer_name, "concluding ", len(cancer_gene6), "genes")

		cancer_gene7 = read_gene_names7(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom7 of cancer_name ", cancer_name, "concluding ", len(cancer_gene7), "genes")

		cancer_gene8 = read_gene_names8(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom8 of cancer_name ", cancer_name, "concluding ", len(cancer_gene8), "genes")

		cancer_gene9 = read_gene_names9(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom9 of cancer_name ", cancer_name, "concluding ", len(cancer_gene9), "genes")

		cancer_gene10 = read_gene_names10(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom10 of cancer_name ", cancer_name, "concluding ", len(cancer_gene10), "genes")

		cancer_gene11 = read_gene_names11(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom11 of cancer_name ", cancer_name, "concluding ", len(cancer_gene11), "genes")

		cancer_gene12 = read_gene_names12(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom12 of cancer_name ", cancer_name, "concluding ", len(cancer_gene12), "genes")

		cancer_gene13 = read_gene_names13(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom13 of cancer_name ", cancer_name, "concluding ", len(cancer_gene13), "genes")

		cancer_gene14 = read_gene_names14(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom14 of cancer_name ", cancer_name, "concluding ", len(cancer_gene14), "genes")

		cancer_gene15 = read_gene_names15(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom15 of cancer_name ", cancer_name, "concluding ", len(cancer_gene15), "genes")

		cancer_gene16 = read_gene_names16(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom16 of cancer_name ", cancer_name, "concluding ", len(cancer_gene16), "genes")

		cancer_gene17 = read_gene_names17(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom17 of cancer_name ", cancer_name, "concluding ", len(cancer_gene17), "genes")

		cancer_gene18 = read_gene_names18(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom18 of cancer_name ", cancer_name, "concluding ", len(cancer_gene18), "genes")

		cancer_gene19 = read_gene_names19(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom19 of cancer_name ", cancer_name, "concluding ", len(cancer_gene19), "genes")

		cancer_gene20 = read_gene_names20(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom20 of cancer_name ", cancer_name, "concluding ", len(cancer_gene20), "genes")

		cancer_gene21 = read_gene_names21(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom21 of cancer_name ", cancer_name, "concluding ", len(cancer_gene21), "genes")

		cancer_gene22 = read_gene_names22(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chrom22 of cancer_name ", cancer_name, "concluding ", len(cancer_gene22), "genes")

		cancer_geneX = read_gene_namesX(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chromX of cancer_name ", cancer_name, "concluding ", len(cancer_geneX), "genes")

		cancer_geneY = read_gene_namesY(cancer_directory_path, cancer_name) 
		os.chdir(current_path) 
		print(" The chromY of cancer_name ", cancer_name, "concluding ", len(cancer_geneY), "genes")

if __name__ == "__main__":
	main()
