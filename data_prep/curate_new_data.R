# Curate new examples from WildCam Gorongosa and Wild Dog Risk Study

library(tidyverse)
library(jsonlite)
library(clipr) #for copy to clipboard



# TrapTagger --------------------------------------------------------------

## Read in Trap Tagger results ---------------------------------------------

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


## Summarize TrapTagger ----------------------------------------------------

tt_season7 %>% count(`Survey Name`)
tt_season7 %>% count(`Species`)
tt_season7 %>% count(`Site Name`)

#summarize
s7_summary <- tt_season7 %>% 
  group_by(Species) %>% 
  summarise(n_images = n(),
            n_sites = n_distinct(`Site Name`))

s7_summary %>% print(n = 100)



## QC ----------------------------------------------------------------------

#found a topi?!
# View(tt_season7 %>% filter(Species == 'Topi'))
#do these images have any other tags?
tt_season7 %>% filter(File == 'Gorongosa_S7_part3/Gorongosa_S7_part3/I12/I12_R1/Correct_05310215.JPG')
tt_season7 %>% filter(File == 'Gorongosa_S7_part3/Gorongosa_S7_part3/I12/I12_R1/Correct_05310216.JPG')
#just the one
#I checked and it's actually a waterbuck; will let the team know to change it

#potentially check:
#blue duiker 
tt_season7 %>% filter(Species == 'Blue Duiker') %>% distinct(`Site Name`) #all from E02 and K10
tt_season7 %>% filter(Species == 'Blue Duiker') %>% distinct(File) %>% print(n = 100)
#the E02 ones are actually impala
write_clip(tt_season7 %>% filter(Species == 'Blue Duiker', `Site Name` == 'E02') %>% 
             select(`Sighting ID`, Timestamp, `Image ID`, File, `Cluster ID`, `Site Name`, Species, `Survey Name`))
#K10 might be blue duiker but this site has a mix of suni and blue duiker in the same days and even same images
View(tt_season7 %>% filter(`Site Name` == 'K10'))
tt_season7 %>% filter(File == 'Gorongosa_S7_part5/Gorongosa_S7_part5/K10/K10_R1/K10_R1/04180152.JPG') %>% pull(Species) #common duiker
tt_season7 %>% filter(File == 'Gorongosa_S7_part5/Gorongosa_S7_part5/K10/K10_R1/K10_R1/04180153.JPG') %>% pull(Species) #common duiker
tt_season7 %>% filter(File == 'Gorongosa_S7_part5/Gorongosa_S7_part5/K10/K10_R1/K10_R1/04180154.JPG') %>% pull(Species) #suni
tt_season7 %>% filter(File == 'Gorongosa_S7_part5/Gorongosa_S7_part5/K10/K10_R1/K10_R1/04180155.JPG') %>% pull(Species) #suni
tt_season7 %>% filter(File == 'Gorongosa_S7_part5/Gorongosa_S7_part5/K10/K10_R1/K10_R1/04180156.JPG') %>% pull(Species) #blue duiker 20 min later

#suni
tt_season7 %>% filter(Species == 'Suni') %>% distinct(`Site Name`) 
#only at K10. I made a git issue. Prob want to check all suni and duikers... 

#qc some sequences with >1 species
sequences_qc <- tt_season7 %>% group_by(`Cluster ID`) %>% 
  filter(Species != 'None') %>% 
  summarise(n_species = n_distinct(Species),
            species_combo = paste(sort(unique(Species)), collapse = ", ")) %>% 
  filter(n_species > 1)

#view combos
sort(unique(sequences_qc$species_combo))

#inspect some suspect ones
(cluster_ids <- sequences_qc %>% filter(species_combo == 'Baboon, Civet') %>% pull(`Cluster ID`)) #cluster 151816718
tt_season7 %>% filter(`Cluster ID` %in% cluster_ids) %>% select(File, Timestamp, Species) %>% print(n=100)
write_clip(tt_season7 %>% filter(`Cluster ID` %in% cluster_ids) %>% select(File, Left, Right, Top, Bottom, Species))


## Summarize by split -------------------------------------------------------

#create site key
holdout_sites <- c("d05", "d03", "g02", "e02", "e06", "f05", "i10", "i04", "i08", "d07", "b05", "g08")
holdout_sites <- toupper(holdout_sites)

site_key <- data.frame('site' = unique(tt_season7$`Site Name`))
site_key <- site_key %>% 
  mutate(split = if_else(site %in% holdout_sites, 'holdout', 'train'))
site_key  
  
#join with tags
tt_season7 <- tt_season7 %>% 
  left_join(site_key, by = c(`Site Name` = 'site'))

#summarize species/split
s7_split_summary <- tt_season7 %>% 
  count(Species, split) %>% 
  pivot_wider(names_from = split,
              values_from = n,
              values_fill = 0)

#view (excluding some sp)
s7_split_summary %>% filter(!Species %in% c('Antelope','Mongoose','Primate',
                                            'Small & Medium Mammals','Topi','Unknown',
                                            'Vehicles/Humans/Livestock',
                                            'None'
                                            )) %>% 
  select(Species, train, holdout) %>% arrange(tolower(Species)) %>% print(n=100)


# Review blanks -----------------------------------------------------------
tt_season7 %>% filter(Species %in% 'None') %>% 
  select(`Site Name`, Left, Right, Top, Bottom, Species, Timestamp, File) %>% nrow()
  #168961

#find ones with non-zero bbox dimensions
tt_season7 %>% filter(Species %in% 'None',
                      Left != 0 | Right != 0 | Top != 0 | Bottom != 0) %>% 
  select(`Site Name`, Left, Right, Top, Bottom, Species, Timestamp, File) %>% print(n=10) %>% nrow 
  #10681

write_clip(tt_season7 %>% filter(Species %in% 'None',
                                 Left != 0 | Right != 0 | Top != 0 | Bottom != 0) %>% 
             select(`Site Name`, Left, Right, Top, Bottom, Species, Timestamp, File) %>% slice(1:5))


