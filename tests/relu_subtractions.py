import numpy as np

# this script is supposed to show that
# | y -x | > | max(0,y) - max(0,x) |
# for all x and y

# make a bunch of x's and y's
n = int(1e7)
x = np.random.uniform(-10, 10, n)
y = np.random.uniform(-10, 10, n)

# find the max
max_x = np.maximum(x, np.zeros(n))
max_y = np.maximum(x, np.zeros(n))
abs_max = np.abs(max_y - max_x)
abs_minus = np.abs(y - x) 
greater = (abs_minus > abs_max)
#print(greater)
all_greater = np.all(greater)

print('| y -x | > | max(0,y) - max(0,x) |,   for all x,y?')
print(all_greater)
