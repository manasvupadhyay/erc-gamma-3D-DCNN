from TSM import * # greedy,smallest_increase,read_all,find_closest_path
nodes = read_all('resources/sehir_xy')


'''
smallest_book, distance_result = smallest_increase(nodes, start_index=45, end_index=51, plot=True, plot_annotate=True)
greedy_book, distance_result = greedy(nodes, start_index=45, end_index=51, plot=False)
find_closest_path(greedy_book, 12, 5, plot=True)
# find_closest_path(smallest_book, 12, 5, plot=True, show_route=True, suptitle='Smallest Increase Algorithm Route')
'''

greedy_book, distance_result = greedy(nodes, start_index=0, end_index=4984, plot=False)

print(distance_result)

inputFile = np.loadtxt('resources/sehir_xy')

i = 0
totDistance = 0.
for i in range(0,len(inputFile)-1):
    index1 = greedy_book[i]
    index2 = greedy_book[i+1]
    a = inputFile[index1]
    b = inputFile[index2]
    totDistance += np.linalg.norm(a-b)
    
print("Total distance (pixel) :", totDistance)
print("Total precipitate :     ", len(inputFile))
print("Mean distance (pixel) : ", totDistance/len(inputFile))
