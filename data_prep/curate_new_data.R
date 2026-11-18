# Curate new examples from WildCam Gorongosa and Wild Dog Risk Study

library(tidyverse)
library(jsonlite)
library(clipr) #for copy to clipboard
library(ggplot2)


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
                                            'Vehicles/Humans/Livestock'
                                            # 'None' #cd use these as background?
                                            )) %>% 
  select(Species, train, holdout) %>% arrange(tolower(Species)) %>% print(n=100)


## Review blanks -----------------------------------------------------------
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


## Compare with v0 ---------------------------------------------------------

#read in v0 dataset size
v0_data <- read_csv('../training_holdout_size_v0.csv')

#specify class re-mapping
class_mapping <- c(
  "bird"= "bird_other",
  "blue duiker"= "duiker_blue",
  "cape buffalo"= "buffalo",
  "thicktailed bushbaby"= "bushbaby",
  "common duiker"= "duiker_common",
  "red duiker"= "duiker_red",
  "honey badger"= "honey_badger",
  "banded mongoose"= "mongoose_banded",
  "egyptian or large grey mongoose"= "mongoose_large_grey",
  "marsh mongoose"= "mongoose_marsh",
  "slender mongoose"= "mongoose_slender",
  "white-tailed mongoose"= "mongoose_white_tailed",
  "bushy-tailed mongoose"= "mongoose_bushy_tailed",
  "rat" = "rodent",
  "sable"= "sable_antelope",
  "samango monkey"= "samango_monkey",
  "vervet monkey"= "vervet_monkey",
  "wild dog"= "wild_dog"
)

#convert to lowercase and match
tt_season7 <- tt_season7 %>% 
  mutate(species_lower = tolower(Species),
         species_mapped = coalesce(class_mapping[species_lower], species_lower))

  #preview
  tt_season7 %>% select(Species, species_lower, species_mapped)

  #view mapping combos
  tt_season7 %>% count(Species, species_mapped) %>% arrange(-n) %>% print(n=100)
  
  #any NAs?
  table(tt_season7$species_mapped, useNA = 'a') #good, none

#did I get them all?
unique(tt_season7$species_mapped)[!(unique(tt_season7$species_mapped) %in% unique(v0_data$species))] 
  #these exist in tt but not v0:
    #-none ✓
    #-vehicles/humans/livestock ✓
    #-unknown ✓
    #-primate ✓
    #-small & medium mammals ✓
    #-mongoose ✓
    #-bushy-tailed mongoose *this didn't map correctly [ok fixed now]
    #-wild_dog ✓ (new class)
    #-duiker_blue ✓ (new class)
    #-topi ✓ (will be removed)
    #-suni ✓ (new class)

unique(v0_data$species)[!(unique(v0_data$species) %in% unique(tt_season7$species_mapped))] 
  #these exist in v0 but not tt
    #-guineafowl_helmeted ✓ (birds not tagged)
    #-duiker ✓ (good, they're all to species I guess)
    #-ground_hornbill ✓ (birds not tagged)
    #-guineafowl_crested ✓ (birds not tagged)
    #-mongoose_dwarf ✓ (no tags in S7)
    #-guineafowl ✓ (birds not tagged)

#redo summary with new classes
s7_split_summary_mapped <- tt_season7 %>% 
  count(species_mapped, split) %>% 
  pivot_wider(names_from = split,
              values_from = n,
              values_fill = 0) %>% 
  rename(training = train)

#convert to long
s7_split_summary_mapped_long <- s7_split_summary_mapped %>% 
  pivot_longer(cols = c('holdout','training'),
               names_to = 'split',
               values_to = 's7') %>% 
  rename(species = species_mapped)

v0_data_long <- v0_data %>% 
  select(-total) %>% 
  pivot_longer(cols = c('holdout','training'),
               names_to = 'split',
               values_to = 'v0')

#join and convert to long again
data_joined <- s7_split_summary_mapped_long %>% 
  full_join(v0_data_long, by = c('species','split'))

data_joined_long <- data_joined %>% 
  pivot_longer(cols = c('s7','v0'),
               names_to = 'source',
               values_to = 'n')

#negate holdout for plotting
# data_joined_long <- data_joined_long %>% 
  # mutate(n = if_else(split == 'holdout', -n, n))

#convert NAs to 0s
data_joined %>% filter(is.na(s7) | is.na(v0)) %>% print(n=50)
data_joined_long %>% filter(is.na(n)) %>% print(n=50)
data_joined_long <- data_joined_long %>% 
  replace_na(list(n = 0))

#format for plotting
data_joined_long_plot <- data_joined_long %>% 
  #reorder species based on total
  mutate(species_ordered = fct_reorder(species, n, .fun = sum)) %>% 
  #convert holdout to negative for diverging plot
  mutate(n = if_else(split == 'holdout', -n, n)) %>% 
  #filter out some classes we won't use
  filter(!species %in% c('none',
                         'antelope',
                         'duiker',
                         'guineafowl',
                         'mongoose',
                         'primate',
                         'small & medium mammals',
                         'topi',
                         'unknown',
                         'vehicles/humans/livestock')) 

# #find breaks for splitting into facets since axis range is huge
# totals <- data_joined %>% 
#   filter(split == 'training') %>% #assign breaks based on training (shd always be higher than holdout)
#   replace_na(list(s7 = 0, v0 = 0)) %>% 
#   mutate(total = s7 + v0) 
# breaks <- classInt::classIntervals(totals %>% pull(total), n = 4, style = 'jenks')
# (break_points <- breaks$brks)
# 
# #assign each species to a break
# totals <- totals %>% 
#   mutate(tier = cut(total, 
#                     breaks = break_points, 
#                     include.lowest = TRUE,
#                     labels = c('one','two','three','four')))
# totals %>% select(species, total, tier) %>% arrange(total) %>% print(n=100)
# 
# #join back to plotting df
# data_joined_long_plot <- data_joined_long_plot %>% 
#   left_join(totals %>% select(species, tier), by = 'species')
  
#get total training size per species
totals <- data_joined %>% 
  filter(split == 'training') %>%
  replace_na(list(s7 = 0, v0 = 0)) %>% 
  mutate(training_total = s7 + v0) 

#and join back with data
data_joined_long_plot <- data_joined_long_plot %>% 
  left_join(totals %>% select(species, training_total), by = 'species') 
  summary(data_joined_long_plot$training_total)

#assign tiers manually
data_joined_long_plot <- data_joined_long_plot %>% 
  mutate(tier = case_when(training_total >= 2755 ~ 'A: very high',
                          training_total < 2755 & training_total >= 277 ~ 'B: high',
                          training_total < 277 & training_total >= 85 ~ 'C: medium',
                          training_total < 85 ~ 'D: low'))
data_joined_long_plot %>% group_by(tier) %>% summarise(n=n_distinct(species))

#plot
splits <- data_joined_long_plot %>% 
  # filter(species_ordered %in% levels(species_ordered)[20:30]) %>% 
  ggplot(aes(x = species_ordered, y = n, fill = source)) +
    geom_bar(stat = 'identity', position = 'stack') +
    # annotate('text', x = 2, y = 40000, label = 'Training', hjust = 1, fontface = 'bold') +
    # annotate('text', x = 2, y = -10000, label = 'Holdout', hjust = 1, fontface = 'bold') +
    geom_hline(yintercept = 0, color = 'black', linewidth = 0.8) +
    scale_y_continuous(labels = abs) +
    coord_flip(
      # ylim = c(-300,300) #to zoom
      ) +
    # facet_wrap(~tier, scales = 'free', ncol = 2) + #comment out for "_full"
    labs(x = NULL, y = 'Number of examples', 
         title = 'Left: holdout\nRight: training') +
    theme_bw(base_size = 14) + theme(legend.position = 'top')
splits

#check a few
# data_joined_long_plot %>% filter(species == 'duiker_blue')

#save
# ggsave('../data_size_facet.png', splits, width = 8, height = 6)
ggsave('../data_size_full.png', splits, width = 6, height = 8)


## Try some scenarios ------------------------------------------------------

#S1: don't add any waterbuck, baboon, impala, warthog, bushbuck from S7
data_joined_long_plot_s1 <- data_joined_long_plot %>% 
  filter(!(source == 's7' & species %in% c('waterbuck','baboon','impala','warthog','bushbuck')))

splits_s1 <- data_joined_long_plot_s1 %>% 
  ggplot(aes(x = species_ordered, y = n, fill = source)) +
  geom_bar(stat = 'identity', position = 'stack') +
  geom_hline(yintercept = 0, color = 'black', linewidth = 0.8) +
  scale_y_continuous(labels = abs) +
  coord_flip() +
  # facet_wrap(~tier, scales = 'free', ncol = 2) +
  labs(x = NULL, y = 'Number of examples', 
       title = 'Left: holdout\nRight: training') +
  theme_bw(base_size = 14) + theme(legend.position = 'top')
splits_s1

ggsave('../data_size_scenario1.png', splits_s1, width = 6, height = 8)

#S2: don't add any waterbuck, baboon, impala, warthog, bushbuck from S7 AND cap all at 20k
data_joined_long_plot_s2 <- data_joined_long_plot %>% 
  filter(!(source == 's7' & species %in% c('waterbuck','baboon','impala','warthog','bushbuck'))) %>% 
  mutate(n = if_else(species %in% c('waterbuck','baboon','impala','warthog','bushbuck') &
                       n > 20000,
                     20000,
                     n),
         n = if_else(species %in% c('waterbuck','baboon','impala','warthog','bushbuck') &
                       n < -20000,
                     -20000,
                     n))

splits_s2 <- data_joined_long_plot_s2 %>% 
  ggplot(aes(x = species_ordered, y = n, fill = source)) +
  geom_bar(stat = 'identity', position = 'stack') +
  geom_hline(yintercept = 0, color = 'black', linewidth = 0.8) +
  scale_y_continuous(labels = abs) +
  coord_flip() +
  # facet_wrap(~tier, scales = 'free', ncol = 2) +
  labs(x = NULL, y = 'Number of examples', 
       title = 'Left: holdout\nRight: training') +
  theme_bw(base_size = 14) + theme(legend.position = 'top')
splits_s2

ggsave('../data_size_scenario2.png', splits_s2, width = 6, height = 8)

#S3: don't add any waterbuck, baboon, impala, warthog, bushbuck from S7 AND cap all at 10k
data_joined_long_plot_s3 <- data_joined_long_plot %>% 
  filter(!(source == 's7' & species %in% c('waterbuck','baboon','impala','warthog','bushbuck'))) %>% 
  mutate(n = if_else(species %in% c('waterbuck','baboon','impala','warthog','bushbuck') &
                       n > 10000,
                     10000,
                     n),
         n = if_else(species %in% c('waterbuck','baboon','impala','warthog','bushbuck') &
                       n < -10000,
                     -10000,
                     n))

splits_s3 <- data_joined_long_plot_s3 %>% 
  ggplot(aes(x = species_ordered, y = n, fill = source)) +
  geom_bar(stat = 'identity', position = 'stack') +
  geom_hline(yintercept = 0, color = 'black', linewidth = 0.8) +
  scale_y_continuous(labels = abs) +
  coord_flip() +
  # facet_wrap(~tier, scales = 'free', ncol = 2) +
  labs(x = NULL, y = 'Number of examples', 
       title = 'Left: holdout\nRight: training') +
  theme_bw(base_size = 14) + theme(legend.position = 'top')
splits_s3

ggsave('../data_size_scenario3.png', splits_s3, width = 6, height = 8)

#S4: also don't add any oribi, nyala, elephant from S7 AND cap all at 5k (ideally cd add them and then cap)
data_joined_long_plot_s4 <- data_joined_long_plot %>% 
  filter(!(source == 's7' & species %in% c('waterbuck','baboon','impala','warthog','bushbuck','oribi','nyala','elephant'))) %>% 
  mutate(n = if_else(species %in% c('waterbuck','baboon','impala','warthog','bushbuck','oribi','nyala','elephant') &
                       n > 5000,
                     5000,
                     n),
         n = if_else(species %in% c('waterbuck','baboon','impala','warthog','bushbuck','oribi','nyala','elephant') &
                       n < -5000,
                     -5000,
                     n))

  #assign new tiers
  data_joined_long_plot_s4 <- data_joined_long_plot_s4 %>% 
    mutate(tier4 = case_when(training_total >= 500 ~ 'A: high',
                            training_total < 500 ~ 'B: low'))
  data_joined_long_plot_s4 %>% group_by(tier4) %>% summarise(n=n_distinct(species))

splits_s4 <- data_joined_long_plot_s4 %>% 
  ggplot(aes(x = species_ordered, y = n, fill = source)) +
  geom_bar(stat = 'identity', position = 'stack') +
  geom_hline(yintercept = 0, color = 'black', linewidth = 0.8) +
  scale_y_continuous(labels = abs) +
  coord_flip() +
  facet_wrap(~tier4, scales = 'free', ncol = 2) +
  labs(x = NULL, y = 'Number of examples', 
       title = 'Left: holdout\nRight: training') +
  theme_bw(base_size = 14) + theme(legend.position = 'top')
splits_s4

ggsave('../data_size_scenario4.png', splits_s4, width = 6, height = 8)
ggsave('../data_size_scenario4_facet.png', splits_s4, width = 8, height = 6)
