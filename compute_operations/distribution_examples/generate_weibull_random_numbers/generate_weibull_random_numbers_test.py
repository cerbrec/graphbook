import json
import time
import matplotlib.pyplot as plt

start = time.time()
# Load the JSON file
with open(r'C:\Users\HP\Documents\Cerberic\weibull-tensor-test.json', 'r') as file:
    decimal_list = json.load(file)

#Group them by intervals of 0.01
dec_dict = {}
for idx in range(0,267):
    lst = [num for num in decimal_list if num >= 0.01 * (idx) and num < 0.01 * (idx+1)]
    dec_dict[str(idx)] = lst 
    
#get min and max values
print(max(decimal_list), min(decimal_list))
plot_dict = {
    # k = 2
    'categories': [f"{i/100:.2f}-{(i+1)/100:.2f}" for i in range(0, 267)],
    'density': [len(dec_dict[key])/1000000 for key in dec_dict]
}

    
end = time.time()
print((end-start) * 1000)


# Plot grouped values
plt.bar(plot_dict["categories"], plot_dict["density"])
plt.xlabel('Categories')
plt.ylabel('Density')
plt.title('Bar Chart of 1000x1000 Tensor')
plt.show()
