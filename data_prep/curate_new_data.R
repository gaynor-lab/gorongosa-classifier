# Curate new examples from WildCam Gorongosa and Wild Dog Risk Study

library(tidyverse)
library(jsonlite)


# Read in Trap Tagger results ---------------------------------------------

#JSON is easy to download (COCO format) but not easy to get quick summaries
# tt_s7p1 <- fromJSON('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part1_Classification_Gorongosa_S7_20261104.json', simplifyVector = TRUE)
# tt_s7p2 <- fromJSON('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part2_Gorongosa_S7_part2_20261104.json', simplifyVector = TRUE)
# tt_s7p3 <- fromJSON('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part3_Classification_Gorongosa_S7_part3_20261104.json', simplifyVector = TRUE)
# tt_s7p4 <- fromJSON('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part4_Classification_Gorongosa_S7_part4_20261104.json', simplifyVector = TRUE)
# tt_s7p5 <- fromJSON('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part5_Gorongosa_S7_part5_20261104.json', simplifyVector = TRUE)
# 
# head(tt_s7p1$images)
# head(tt_s7p1$annotations)

#Instead, use custom CSV with row=Sighting. Critical columns are File, Timestamp, Species, Camera Name, Survey Name
tt_s7p1 <- read_csv('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part1_Classification_Gorongosa_S7_20261104.csv')
tt_s7p2 <- read_csv('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part2_Gorongosa_S7_part2_20261104.csv')
tt_s7p3 <- read_csv('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part3_Classification_Gorongosa_S7_part3_20261104.csv')
tt_s7p4 <- read_csv('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part4_Classification_Gorongosa_S7_part4_20261104.csv')
tt_s7p5 <- read_csv('/Users/caraappel/Documents/_UBC/gorongosa_classifier/trap_tagger/Gaynor Lab_cappel_Gorongosa_S7_part5_Gorongosa_S7_part5_20261104.csv')

#combine
tt_season7 <- bind_rows(tt_s7p1, tt_s7p2, tt_s7p3, tt_s7p4, tt_s7p5)

head(tt_season7 %>% select(File, Timestamp, Species, `Camera Name`, `Survey Name`))

# Summarize TrapTagger ----------------------------------------------------

tt_season7 %>% count(`Survey Name`)
tt_season7 %>% count(`Species`)
tt_season7 %>% count(`Camera Name`)

#clean up site column
sort(unique(tt_season7$`Camera Name`))
tt_season7 <- tt_season7 %>% 
  mutate(site = str_split_i(`Camera Name`, '-', 1))
sort(unique(tt_season7$site))

#summarize

#found a topi?!
View(tt_season7 %>% filter(Species == 'Topi'))
#do these images have any other tags?
tt_season7 %>% filter(File == 'Gorongosa_S7_part3/Gorongosa_S7_part3/I12/I12_R1/Correct_05310215.JPG')
tt_season7 %>% filter(File == 'Gorongosa_S7_part3/Gorongosa_S7_part3/I12/I12_R1/Correct_05310216.JPG')
#just the one




