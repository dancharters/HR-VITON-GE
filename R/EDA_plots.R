library(tikzDevice)
library(tidyverse)

data_path = "../eda_output_data/"
image_path = "../../report/images/"

# Garment colours -----
garment_colours = read_csv(paste0(data_path, "garment_colours.csv"))

tikz(file = paste0(image_path, "garment_colours.tex"), width=6, height=2.8)
ggplot(garment_colours, aes(x = reorder(Color, Count, FUN = function(x) -max(x)), y = Count, fill = Color)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  scale_fill_identity() +
  labs(x = "Average colour", y = "Number of garments") +
  theme_minimal() +
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.title.x = element_text(margin = margin(t = 5, b = -5)),
        axis.line.y = element_line(color = "#ebebeb", size = 0.5)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, max(garment_colours$Count+200)))
endoffile <- dev.off() 

# Garment types ----
# Elbow plot
garment_clusters = read_csv(paste0(data_path, "garment_clusters_elbow_data.csv"))

tikz(file = paste0(image_path, "garment_clusters_elbow.tex"), width=5, height=2.2)
ggplot(data = garment_clusters, aes(x = k, y = distortion/(10^8))) +
  geom_line(size = 0.8, color = "#0a375f") +
  geom_point(size = 1.2, color = "#0a375f") +
  labs(x = "Number of clusters", y = "Distortion $(10^8)$") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(legend.position = "none") +
  theme_minimal() +
  theme(text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 12)),
        axis.title.x = element_text(margin = margin(t = 7, b = -3))) +
  scale_x_continuous(breaks = c(0, 5, 10, 15, 20, 25, 30)) +
  geom_vline(aes(xintercept = 5), linetype = "dotted", color = "#0a375f", size = 0.8)
endoffile <- dev.off() 

# Cluster plot
garment_types = read_csv(paste0(data_path, "garment_clusters.csv"))
table(garment_types$cluster)
tikz(file = paste0(image_path, "garment_clusters.tex"), width=4, height=2.5)
ggplot(garment_types, aes(x = factor(cluster))) + 
  geom_bar(fill = "#0a375f", width = 0.7) +
  theme_minimal()  +
  scale_x_discrete(breaks = 0:5, labels = 1:6)+
  labs(x = "Garment cluster", y = "Number of garments") +
  theme(axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 12)),
        axis.title.x = element_text(margin = margin(t = 7, b = -3)))
endoffile <- dev.off() 

# Skin colour ----
skin_colours = read_csv(paste0(data_path, "skin_colours.csv"))

# Define a function to calculate the grayscale value for an RGB color
rgb_to_grayscale <- function(rgb) {
  # Remove the parentheses from the string
  rgb <- gsub("[()]", "", rgb)
  # Split the string into substrings
  rgb <- strsplit(rgb, ", ")[[1]]
  # Convert the substrings to numeric values and store them in a list
  rgb <- as.list(as.numeric(rgb))
  return(0.2126 * rgb[[1]] + 0.7152 * rgb[[2]] + 0.0722 * rgb[[3]])
}

# Calculate the average brightness for each color
skin_colours$avg_brightness <- sapply(skin_colours$avg_rgb_color, rgb_to_grayscale)

tikz(file = paste0(image_path, "skin_colours.tex"), width=6, height=2.5)
ggplot(skin_colours, aes(x = reorder(avg_hex_color, avg_brightness, FUN = function(x) -max(x)), y = num_images, fill = avg_hex_color)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  scale_fill_identity() +
  labs(x = "Average skin colour", y = "Number of images") +
  theme_minimal() +
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.title.x = element_text(margin = margin(t = 5, b = -5)),
        axis.line.y = element_line(color = "#ebebeb", size = 0.5)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, max(skin_colours$num_images+100)))
endoffile <- dev.off() 


# Hair colour ----
hair_colours = read_csv(paste0(data_path, "hair_colours.csv"))

# Calculate the average brightness for each color
hair_colours$avg_brightness <- sapply(hair_colours$avg_rgb_color, rgb_to_grayscale)

tikz(file = paste0(image_path, "hair_colours.tex"), width=6, height=2.5)
ggplot(hair_colours, aes(x = reorder(avg_hex_color, avg_brightness, FUN = function(x) -max(x)), y = num_images, fill = avg_hex_color)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  scale_fill_identity() +
  labs(x = "Average hair colour", y = "Number of images") +
  theme_minimal() +
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.title.x = element_text(margin = margin(t = 5, b = -5)),
        axis.line.y = element_line(color = "#ebebeb", size = 0.5)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, max(hair_colours$num_images+100)))
endoffile <- dev.off() 

# Pose types ----
# Elbow plot
pose_clusters = read_csv(paste0(data_path, "pose_clusters_elbow_data.csv"))

tikz(file = paste0(image_path, "pose_clusters_elbow.tex"), width=5, height=2.5)
ggplot(data = pose_clusters, aes(x = k, y = distortion/(10^8))) +
  geom_line(size = 0.8, color = "#0a375f") +
  geom_point(size = 1.2, color = "#0a375f") +
  labs(x = "Number of clusters", y = "Distortion $(10^8)$") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(legend.position = "none") +
  theme_minimal() +
  theme(text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 12)),
        axis.title.x = element_text(margin = margin(t = 7, b = -3))) +
  scale_x_continuous(breaks = c(0, 5, 10, 15, 20, 25, 30)) +
  geom_vline(aes(xintercept = 4), linetype = "dotted", color = "#0a375f", size = 0.8)
endoffile <- dev.off() 

# Cluster plot
pose_types = read_csv(paste0(data_path, "pose_clusters.csv"))
table(pose_types$Cluster)
tikz(file = paste0(image_path, "pose_clusters.tex"), width=4, height=2.5)
ggplot(pose_types, aes(x = factor(Cluster))) + 
  geom_bar(fill = "#0a375f", width = 0.6) +
  theme_minimal()  +
  scale_x_discrete(breaks = 0:5, labels = 1:6)+
  labs(x = "Pose cluster", y = "Number of images") +
  theme(axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 12)),
        axis.title.x = element_text(margin = margin(t = 7, b = -3)))+
  scale_y_continuous(expand = c(0, 0), limits = c(0, 4000))
endoffile <- dev.off() 

# Image quality ----

# Women images 
women_images = read_csv(paste0(data_path, "women_images_stats.csv"))
mean(women_images$brightness)
mean(women_images$blurriness)
mean(women_images$entropy)

# Set seed for reproducibility
set.seed(240898)

# Sample 1000 rows from the dataframe
women_images_sample <- women_images[sample(nrow(women_images), 1000), ]

# Create the scatterplot
tikz(file = paste0(image_path, "women_images_stats.tex"), width=6, height=3.3)
ggplot(women_images_sample, aes(x = blurriness, y = entropy, size = brightness)) +
  geom_point(alpha = 0.7, color = "#0a375f") +
  scale_size_continuous(range = c(4*(0.2),4*(0.9))) +
  theme_minimal() +
  labs(x = "Blurriness",
       y = "Entropy",
       size = "Brightness") +
  theme(legend.position = "bottom",
        text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 12)),
        axis.title.x = element_text(margin = margin(t = 7, b = -5))) +
  guides(size = guide_legend(nrow = 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(5.5, 9)) +
  scale_x_continuous(expand = c(0, 0), limits = c(30, 95))
endoffile <- dev.off() 

# Garment images 
garment_images = read_csv(paste0(data_path, "garment_images_stats.csv"))
mean(garment_images$brightness)
mean(garment_images$blurriness)
mean(garment_images$entropy)

# Set seed for reproducibility
set.seed(240898)

# Sample 1000 rows from the dataframe
garment_images_sample <- garment_images[sample(nrow(garment_images), 1000), ]

# Create the scatterplot
tikz(file = paste0(image_path, "garment_images_stats.tex"), width=6, height=3.3)
ggplot(garment_images_sample, aes(x = blurriness, y = entropy, size = brightness)) +
  geom_point(alpha = 0.7, color = "#0a375f") +
  scale_size_continuous(range = c(4*(0.2),4*(0.9))) +
  theme_minimal() +
  labs(x = "Blurriness",
       y = "Entropy",
       size = "Brightness") +
  theme(legend.position = "bottom",
        text = element_text(size = 10),
        axis.title = element_text(size = 10),
        axis.title.y = element_text(margin = margin(r = 12)),
        axis.title.x = element_text(margin = margin(t = 7, b = -5))) +
  guides(size = guide_legend(nrow = 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(3, 7.2)) +
  scale_x_continuous(expand = c(0, 0), limits = c(30, 112))
endoffile <- dev.off() 