import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib as mpl
import matplotlib.colors
import random

alpha_dict = {' ':0,'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
words = input('Say something!   ')
flatten = words.lower()

numbers = []
for x in flatten:
    if x in alpha_dict:
        num = alpha_dict[x]
        numbers.append(num)
    else:
        numbers.append(random.randint(26,100))
print(numbers)

# first character chooses plot maybe? 

a = numbers[0]
b = None
c = None
e = None
f = None


if a%5 == 0:
    ax = plt.figure().add_subplot(projection='3d')

    theta = np.linspace(-4*np.pi, 24*np.pi, 100)

    z = np.linspace(-2,2,100)
    r = z**2+1
    x = r*np.sin(theta)
    y = r*np.cos(theta)
    y2 = -r*np.cos(theta)

    ax.plot(x,y,z,'b',label = 'random curve 1',)
    ax.plot(x,y2,z,'y',label = 'random curve 2',)

    # plt.show()
    # mpl.rcParams["savefig.frameon"] = False
    mpl.rcParams["savefig.dpi"] = 2000
    mpl.rcParams["savefig.transparent"] = True
    plt.savefig('my_plot.jpg', format='jpg', pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    
    
    # 'C:\Users\John Fessler\My Python Stuff\personal website 2023\graph\plot_images')

    #to disable interative mode always
    # plt.close()


    # ax.legend()
    # plt.rcParams["figure.figsize"] = [10, 10]
    # plt.rcParams["figure.autolayout"] = True

    # plt.figure()

    # ax.plot(x,y,z,'b',label = 'random curve 1',)
    # ax.plot(x,y2,z,'y',label = 'random curve 2',)

    # img_buf = io.BytesIO()
    # plt.savefig(img_buf, format='png')

    # im = Image.open(img_buf)
    # im.show(title="My Image")

    # img_buf.close()
    

elif a%5 == 1:
# Lorenz attractor
    def lorenz(xyz, *, s=10, r=28, b=2.667):
        x, y, z = xyz
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])
    
    dt = 0.01
    num_steps = 10000
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = (0., 1., 1.05)
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
   
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()
elif a%5 == 2:
# Triangular 3D filled contour
    n_angles = 48
    n_radii = 8
    min_radius = 0.25

    # Create the mesh in polar coordinates and compute x, y, z.
    radii = np.linspace(min_radius, 0.95, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    x = (radii*np.cos(angles)).flatten()
    y = (radii*np.sin(angles)).flatten()
    z = (np.cos(radii)*np.cos(3*angles)).flatten()

    # Create a custom triangulation.
    triang = tri.Triangulation(x, y)

    # Mask off unwanted triangles.
    triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)) < min_radius)

    ax = plt.figure().add_subplot(projection='3d')
    ax.tricontourf(triang, z, cmap=plt.cm.CMRmap)

    # Customize the view angle so it's easier to understand the plot.
    ax.view_init(elev=45.)

    plt.show()

elif a%5 == 3:
# 3d voxel/volumetric plot
    def midpoints(x):
        sl = ()
        for _ in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x

    # prepare some coordinates, and attach rgb values to each
    r, g, b = np.indices((17, 17, 17)) / 16.0
    rc = midpoints(r)
    gc = midpoints(g)
    bc = midpoints(b)

    # define a sphere about [0.5, 0.5, 0.5]
    sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

    # combine the color components
    colors = np.zeros(sphere.shape + (3,))
    colors[..., 0] = rc
    colors[..., 1] = gc
    colors[..., 2] = bc

    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(r, g, b, sphere, facecolors=colors, edgecolors=np.clip(2*colors - 0.5, 0, 1), linewidth=0.5)
    ax.set(xlabel='r', ylabel='g', zlabel='b')
    ax.set_aspect('auto')

    plt.show()
    
elif a%5 == 4:
# triangular 3D surfaces
    fig = plt.figure(figsize=plt.figaspect(0.5))
    u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
    v = np.linspace(-0.5, 0.5, endpoint=True, num=10)
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()

    # This is the Mobius mapping, taking a u, v pair and returning an x, y, z
    # triple
    x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
    y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
    z = 0.5 * v * np.sin(u / 2.0)

    # Triangulate parameter space to determine the triangles
    tri = tri.Triangulation(u, v)

    # Plot the surface.  The triangles in parameter space determine which x, y, z
    # points are connected by an edge.
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
    ax.set_zlim(-1, 1)
    plt.show()
else:
    print("Hello there! No graph implemented for your statement!")

    



# x = [1,2,3]
# y = [-4,5,6]
# # z = [1,2,3]
# # plt.plot(x,y,z)
# plt.plot(x,y)
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# # plt.zlabel('z-axis')
# plt.title('Graphics')