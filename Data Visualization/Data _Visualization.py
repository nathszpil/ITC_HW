import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('Countries.csv')

# Global Statistics

# Population Distribution by Region (Pie Chart)
region_population = data.groupby('Zone')['Population'].sum()
plt.figure(figsize=(8, 6))
plt.pie(region_population, labels=region_population.index, autopct='%1.1f%%', startangle=140)
plt.title('Population Distribution by Region')
plt.show()


# Press Freedom Levels Across Regions (Bar Chart) with improved x-axis labels and ordered legend
press_freedom_by_region = data.groupby('Zone')['Press Freedom'].value_counts().unstack().fillna(0)

legend_order = ['Good Situation', 'Satisfactory Situation', 'Noticeable Problems', 'Difficult Situation', 'Very Serious Situation']

plt.figure(figsize=(12, 6))  # Adjusting the figure size for better visualization
ax = press_freedom_by_region.plot(kind='bar', stacked=True)
plt.xlabel('Region')
plt.ylabel('Count')
plt.title('Press Freedom Levels Across Regions')

# Set x-axis tick positions to align with the bars
ax.set_xticks(range(len(press_freedom_by_region.index)))
ax.set_xticklabels(press_freedom_by_region.index, rotation=45)  # Rotating x-axis labels for better readability

# Reordering legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Press Freedom', title_fontsize='13', labels=legend_order, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Homicide Rate Depending on Zone (Bar Chart)
homicide_by_zone = data.groupby('Zone')['Homicide rate'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
homicide_by_zone.plot(kind='bar', color='red')
plt.xlabel('Zone')
plt.ylabel('Average Homicide Rate')
plt.title('Homicide Rate Depending on Zone')
plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better visibility
plt.show()


# Belgium-specific Statistics

# Filter data for Belgium
belgium_data = data[data['Country'] == 'BELGIUM']

# Create a pie chart for urban vs rural population in Belgium
urban_population = belgium_data['Urban Population (%)'].values[0]
rural_population = 100 - urban_population  # Calculate rural population as the complement of urban population


labels = ['Urban Population', 'Rural Population']
sizes = [urban_population, rural_population]
colors = ['lightblue', 'lightgreen']

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Urban vs Rural Population in Belgium')
plt.show()


# Filter data for Western Europe and append Belgium
western_europe_data = data[data['Zone'] == 'Western Europe']
belgium_data = data[data['Country'] == 'BELGIUM']
western_europe_data = pd.concat([western_europe_data, belgium_data])

# Plotting Belgium vs Western Europe
plt.bar(western_europe_data['Country'], western_europe_data['Homicide rate'], color='blue')
plt.xlabel('Country')
plt.ylabel('Homicide Rate')
plt.title('Belgium vs Western Europe: Homicide Rate')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()