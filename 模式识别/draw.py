import numpy as np 
import matplotlib.pyplot as plt 
 
def draw_curve(para, arange_x, arange_y):
    if len(para) == 6:
        x = np.arange(arange_x[0], arange_x[1], 0.1)
        y = np.arange(arange_y[0], arange_y[1], 0.1)
        x, y = np.meshgrid(x,y)
        plt.contour(x, y, para[0]*x**2 + para[1]*x*y + para[2]*y**2 + para[3]*x + para[4]*y, [para[5]])
        # plt.axis('scaled')
    elif len(para) == 3:
        x = np.arange(arange_x[0], arange_x[1])
        y = (para[2] - para[0] * x) / para[1]
        plt.plot(x, y)
        # plt.axis('scaled')
    plt.xlim(arange_x)
    plt.ylim(arange_y)

def draw_sample(xfs, xms, idx):
    plt.scatter(xms[0], xms[1], s=3, c='red')
    plt.scatter(xfs[0], xfs[1], s=3, c='blue')
    # plt.show()

if __name__ == "__main__":
    a = [0, 0, 0, 1, 1, -1]
    # a = [0, 0, 0, 0.0017334020726680003, 0.0007502622343111623, -0.6921269983704623]
    draw_curve(a, [-10, 10], [-10, 10])
    plt.show()