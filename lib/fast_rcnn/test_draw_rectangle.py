import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

class Annotate(object):
    def __init__(self, image):
        self.ax = plt.axes([0,0,1,1])

        self.rect = Rectangle((0,0), 0, 0, fill=False, edgecolor='b')
        self.im = image
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        #self.ax.patch.set_visible(False)
        #self.ax.figure.set_visible(False)
        #self.ax.figure.set_frameon(False)
        self.ax.axis('off')
        self.ax.imshow(self.im)

        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.button_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.button_release)

        self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)
    def key_press(self, event):
          print '%s is pressed'% event.key
          if event.key == 'w':
            self.ax.figure.savefig('/home/ljyang/Desktop/test.png')#, bbox_inches='tight', pad_inches=0)
    def button_press(self, event):
        print 'press'
        self.x0 = event.xdata
        self.y0 = event.ydata

    def button_release(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        print 'rectangle coordinates: (%.01f, %.01f, %.01f, %.01f)' % (self.x0, self.y0, self.x1, self.y1)
        
        #add new box
        new_x0 = self.x0+30
        new_y0 = self.y0-40
        new_width = 100
        new_height = 70
        self.rect2 = Rectangle((new_x0, new_y0), new_width, new_height, fill=False, edgecolor='g',label='test test')
        self.ax.add_patch(self.rect2)
        self.ax.legend()
        self.ax.figure.canvas.draw()


im_path = '/home/ljyang/work/data/visual_genome/images/1.jpg'
im = cv2.imread(im_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
a = Annotate(im)
plt.show()