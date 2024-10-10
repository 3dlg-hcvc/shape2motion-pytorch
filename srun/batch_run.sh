#!/bin/bash

sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="true",NUM_CHANNELS="6",SEED="1",OUTPUT_FOLDER="results-rgbn-aug",AUGMENTATION="true" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="true",NUM_CHANNELS="6",SEED="2",OUTPUT_FOLDER="results-rgbn-aug",AUGMENTATION="true" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="true",NUM_CHANNELS="6",SEED="3",OUTPUT_FOLDER="results-rgbn-aug",AUGMENTATION="true" s2m_sbatch.sh

# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="false",NUM_CHANNELS="3",SEED="1",OUTPUT_FOLDER="results-rgb-aug",AUGMENTATION="true" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="false",NUM_CHANNELS="3",SEED="2",OUTPUT_FOLDER="results-rgb-aug",AUGMENTATION="true" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="false",NUM_CHANNELS="3",SEED="3",OUTPUT_FOLDER="results-rgb-aug",AUGMENTATION="true" s2m_sbatch.sh

# sbatch --export=ALL,HAS_COLOR="false",HAS_NORMAL="true",NUM_CHANNELS="3",SEED="1",OUTPUT_FOLDER="results-n-aug",AUGMENTATION="true" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="false",HAS_NORMAL="true",NUM_CHANNELS="3",SEED="2",OUTPUT_FOLDER="results-n-aug",AUGMENTATION="true" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="false",HAS_NORMAL="true",NUM_CHANNELS="3",SEED="3",OUTPUT_FOLDER="results-n-aug",AUGMENTATION="true" s2m_sbatch.sh

# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="true",NUM_CHANNELS="6",SEED="1",OUTPUT_FOLDER="results-rgbn",AUGMENTATION="false" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="true",NUM_CHANNELS="6",SEED="2",OUTPUT_FOLDER="results-rgbn",AUGMENTATION="false" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="true",NUM_CHANNELS="6",SEED="3",OUTPUT_FOLDER="results-rgbn",AUGMENTATION="false" s2m_sbatch.sh

# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="false",NUM_CHANNELS="3",SEED="1",OUTPUT_FOLDER="results-rgb",AUGMENTATION="false" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="false",NUM_CHANNELS="3",SEED="2",OUTPUT_FOLDER="results-rgb",AUGMENTATION="false" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="true",HAS_NORMAL="false",NUM_CHANNELS="3",SEED="3",OUTPUT_FOLDER="results-rgb",AUGMENTATION="false" s2m_sbatch.sh

# sbatch --export=ALL,HAS_COLOR="false",HAS_NORMAL="true",NUM_CHANNELS="3",SEED="1",OUTPUT_FOLDER="results-n",AUGMENTATION="false" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="false",HAS_NORMAL="true",NUM_CHANNELS="3",SEED="2",OUTPUT_FOLDER="results-n",AUGMENTATION="false" s2m_sbatch.sh
# sbatch --export=ALL,HAS_COLOR="false",HAS_NORMAL="true",NUM_CHANNELS="3",SEED="3",OUTPUT_FOLDER="results-n",AUGMENTATION="false" s2m_sbatch.sh

