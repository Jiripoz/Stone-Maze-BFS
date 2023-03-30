print("Starting script ...")

from collections import deque
import os 
import json
try:
    import numpy as np
except: 
    print("Numpy is required to run this script.")
    quit()


os.makedirs("epochs", exist_ok=True)

try: 
    INPUT = np.loadtxt('input.txt', dtype = float).astype(int)
except:
    print("Please place the input.txt file in the same folder this script is")
    quit()

print("The solution was made using BFS. It may take a minute or two for the script to run.")

DIRECTIONS = { "U": (0, 1),  "D": (0, -1),  "R": (1, 0),  "L": (-1, 0) }
RULES = {
    #Rules for white cells
    0:{ 0:0, 1:0, 2:1, 3:1, 4:1, 5:0, 6:0, 7:0, 8:0 },
    #Rules for green cells
    1:{ 0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:0, 7:0, 8:0 }
}

def change_cell_state(matrix, i, j):
    if matrix[i][j] == 3 or matrix[i][j] == 4:
        return matrix[i][j]

    count = 0
    for x in range(max(0, i-1), min(i+2, matrix.shape[0])):
        for y in range(max(0, j-1), min(j+2, matrix.shape[1])):
            if x==i and y==j:
                continue
            if matrix[x][y] == 1:
                count+=1

    return int(RULES[int(matrix[i][j])][count])

def create_new_epoch(epoch):
    temp = np.loadtxt(f'epochs/epoch_{epoch-1}.txt', dtype = float).astype(int)
    matrix = np.full((temp.shape[0],temp.shape[1]), 2)
    for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = change_cell_state(temp, i, j)
    np.savetxt(f"epochs/epoch_{epoch}.txt", matrix.astype(int), fmt="%i", delimiter=" ")

    return matrix

def is_outside_boundaries(x, y):
    if x<0 or x>=INPUT.shape[0] or y<0 or y>= INPUT.shape[1]:
        return True
    else: 
        return False

def verify_cell(x, y, epoch):
    if is_outside_boundaries(x, y):
        return False
    # If epoch does not exist, create
    if not os.path.isfile(f"epochs/epoch_{epoch}.txt"):
        matrix = create_new_epoch(epoch)
        if matrix[x][y] == 1:
            return False
        else: return True
    # If it exists, read
    else:
        matrix = np.loadtxt(f"epochs/epoch_{epoch}.txt", dtype=float).astype(int)
        if matrix[x][y] == 1:
            return False
        else: return True

def solve(start, end, initial_matrix):
    np.savetxt(f"epochs/epoch_{0}.txt", initial_matrix.astype(int), fmt="%i", delimiter=" ")

    queue = deque([start])
    visited = set([start])
    prev = {start: None}

    while queue:
        curr = queue.popleft()

        if curr[0] == end[0] and curr[1] == end[1]:
            path= []

            while curr != start:
                path.append(curr)
                curr=prev[curr]
            
            path.append(start)
            path.reverse()
            
            return path
        
        for item in DIRECTIONS:
            dx, dy = DIRECTIONS[item]
            next_cell = (curr[0]+dx, curr[1]+dy, curr[2]+1)

            if (next_cell not in visited and
                verify_cell(next_cell[0], next_cell[1], next_cell[2])):
                queue.append(next_cell)
                visited.add(next_cell)
                prev[next_cell] = curr

    return None


def reverse_map_directions(fx, fy, nx, ny):
    deltax = nx-fx
    deltay = ny-fy
    coordinates = (deltay, deltax)
    for key, value in DIRECTIONS.items():
        if value == coordinates:
            return key
    return None

def map_path_directions(path):
    next_path = path[1:] + [(-1, -1, -1)]
    joined_paths = list(zip(path, next_path))[:-1]

    directions = []
    for current, next in joined_paths:
        cx, cy, _ = current
        nx, ny, _ = next
        directions.append(reverse_map_directions(cx, cy, nx, ny))

    return directions

# x and y refer to points in the matrix, z refers to current epochs
start = (0, 0, 0)
end = (64, 84)

path = solve(start, end, INPUT)
direction_list = np.array(map_path_directions(path))

np.savetxt("solution.txt", direction_list, fmt="%s", newline=" ")

print("Done! You can find the answer in solution.txt. \nYou can also uncomment the code to generate an animated gif for yourself.")
print("I've also uploaded the animation to youtube: https://www.youtube.com/shorts/ppQvA4Ob2l8")

# ~~~~~~~~~~ IF YOU WISH TO GENERATE THE ANIMATED GIF UNCOMMENT THE CODE BELLOW ~~~~~~~~~~ #

# try: 
#     import plotly.express as px
#     import imageio
#     from PIL import Image
#     import matplotlib.pyplot as plt
# except Exception as e:
#     print(f"Missing imports. Please install the required packages.\nError: {e}")

# os.makedirs("epoch_images", exist_ok=True)
# fig = plt.figure()
# ax = fig.add_subplot(111)

# cmap = plt.matplotlib.colors.ListedColormap(['white', '#00af55', "yellow", "yellow", "yellow", "red"])
# fig, ax = plt.subplots(figsize=(5, 5))

# def compress(filepath):
#     image = Image.open(filepath)

#     image.save("image-file-compressed", 
#                  "JPEG", 
#                  optimize = True, 
#                  quality = 1)
#     return

# for m in range(0, 273):
#     matrix = np.loadtxt(f'epochs/epoch_{m}.txt', dtype=float).astype(int)
#     print(f"epoch: {m}, value: {matrix[path[m][0]][path[m][1]]}")
#     matrix[path[m][0]][path[m][1]] = 5
#     x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))

#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.matshow(matrix, cmap=cmap, extent=[0, matrix.shape[1], matrix.shape[0], 0])

#     for i in range(matrix.shape[0]+1):
#         ax.axhline(i, color='black', linewidth=0.5)
#     for j in range(matrix.shape[1]+1):
#         ax.axvline(j, color='black', linewidth=0.5)

#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.tight_layout()
#     fig.savefig(f'epoch_images/epoch_{m}.jpg', dpi=300)
#     compress(f'epoch_images/epoch_{m}.jpg')

#     plt.close(fig)

# image_folder = 'epoch_images/'
# images = []

# for i in range(0,274):
#     images.append(imageio.imread(f"epoch_images/epoch_{i}.jpg"))

# imageio.mimsave('animation.gif', images, fps=7)
