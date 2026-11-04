#Summarize data by species, site, day/night

library(tidyverse)


# Read in data ------------------------------------------------------------
data <- read_csv('/Users/caraappel/Documents/_UBC/gorongosa_classifier/nighttime_110226_partial.csv',
                 col_types = cols(error = col_character()))
  problems(data)
  head(data)
  
  #Errors?
  table(data$error, useNA = 'a')

#Remove that row
data <- data %>% 
  filter(is.na(error))



# Parse -------------------------------------------------------------------
data_parsed <- data %>% 
  mutate(image_name = gsub('/scratch/st-kgaynor-1/gorongosa_classifier/training_data/', '', image_path)) %>% 
  mutate(image_base = str_split_i(image_name, '.jp', 1)) %>% 
  separate_wider_delim(image_base, delim = '_', 
                       names = c('basename','site','species'),
                       too_few = 'debug',
                       too_many = 'debug')


#Debug
table(data$image_base_ok, useNA = 'a')
View(data %>% filter(image_base_ok == 'FALSE'))
  #some have too many: e.g., if species is 'honey_badger' or image basename is '01010036_02'
  #some have too few: e.g., entire filename is '01010064.JPG'

#Look for string matches explicitly
data_parsed <- data %>% 
  mutate(image_name = gsub('/scratch/st-kgaynor-1/gorongosa_classifier/training_data/', '', image_path)) %>% 
  mutate(image_base = str_split_i(image_name, '.jp', 1)) %>% 
  extract(image_base,
          into = c("site", "species"),
          regex = ".*_([[:alpha:]]\\d+)_(.+)$",
          remove = FALSE)
  
#QC site
sort(unique(data_parsed$site, useNA = 'a'))
View(data_parsed %>% filter(is.na(site))) #yes, these 191 truly have no site (why?)

#remove them
data_parsed <- data_parsed %>% 
  filter(!is.na(site))

#QC species
sort(unique(data_parsed$species, useNA = 'a'))
View(data_parsed %>% filter(is.na(species))) #good, none without species


# Summarize grayscale ------------------------------------------------------

hist(data_parsed$gray_scale_fraction)

#Use a cutoff of 0.5
data_parsed <- data_parsed %>% 
  mutate(night = if_else(gray_scale_fraction >= 0.5, TRUE, FALSE))

#But inspect some edge cases
data_parsed %>% 
  filter(gray_scale_fraction > 0.4 &
           gray_scale_fraction < 0.6) %>% 
  pull(image_name)

  #Some are totally black (malfunction or animal in front of the camera?)
  #Others are over-exposed or odd-colored
  #I imagine lots of these will be removed after QCing the crops -- check again


# Summarize species/site/night --------------------------------------------

summary <- data_parsed %>% 
  group_by(species) %>% 
  summarise(n_images = n(),
            n_sites = n_distinct(site),
            # n_night = sum(night),
            prop_night = sum(night) / n_images)

summary %>% print(n = 50)

#remember this is before splits, so the distributions in train/test will look different


