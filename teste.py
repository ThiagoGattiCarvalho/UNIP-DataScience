import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Generate sample data for three groups
np.random.seed(0)
group1 = np.random.normal(loc=10, scale=1, size=30)
group2 = np.random.normal(loc=12, scale=1, size=30)
group3 = np.random.normal(loc=11, scale=1, size=30)

# Perform ANOVA test
f_statistic, p_value = f_oneway(group1, group2, group3)

# Print ANOVA results
print("ANOVA F-statistic:", f_statistic)
print("ANOVA p-value:", p_value)

# Plot ANOVA chart
plt.figure(figsize=(8, 6))
plt.boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('ANOVA Chart')
plt.xlabel('Groups')
plt.ylabel('Values')
plt.grid(axis='y')
plt.show()


