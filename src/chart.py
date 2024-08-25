# -*- coding: utf-8 -*-

# Import libraries
import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

mlp.use("Qt5Agg")

"""
This module contains the classes InteractivePlot and InteractiveScatter. The
charts facilitate pop-up annotations.
"""

class Line():
    def __init__(self,
                 x_points,
                 y_points,
                 c,
                 label):
        self.x_points = x_points
        self.y_points = y_points
        self.c = c
        self.label = label
        

class InteractivePlot:
    def __init__(self, name):
        self.name = name
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
#        self.ax.set_yscale('log')
 #       self.ax.set_xscale('log')
        self.ax.set_yscale('linear')
        self.ax.set_xscale('linear')
        self.cmap = cm.get_cmap("Spectral")
        self.lines = []
        
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        
        
    def add_line(self, x_points, y_points, c, label):
        self.lines.append(Line(x_points=x_points,
                               y_points=y_points,
                               c=c,
                               label=label))
            
        
    
    def update_annot(self, l, idx):
        posx, posy = [l.get_xdata()[idx], l.get_ydata()[idx]]
        self.annot.xy = (posx, posy)
        text = f'{l.get_label()}'
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            
            for line in self.lines:
                cont, ind = line.l.contains(event)
                if cont:
                    self.update_annot(line.l, ind['ind'][0])
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                else:
                    if vis:
                        self.annot.set_visible(False)
                        self.fig.canvas.draw_idle()
    
    
    def render(self):
        min_c = min(self.lines,key=lambda x:x.c).c
        max_c = max(self.lines,key=lambda x:x.c).c
        self.norm = colors.Normalize(min_c, max_c)
        for line in self.lines:
            line.l, = self.ax.plot(line.x_points,
                                   line.y_points,
                                   color=self.cmap(self.norm(line.c)),
                                   label=line.label)
    
    
    def show(self):
        self.render()
        self.fig.colorbar(cm.ScalarMappable(norm=self.norm, cmap=self.cmap), ax=self.ax)
        self.ax.set_title(self.name)
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.show()
    
    def savefig(self, path, bbox_inches="tight", dpi=240):
        self.fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)
    
    def close(self):
        plt.close(self.fig)





        

        
        
class InteractiveScatter:
    def __init__(self, name):
        self.name = name
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
#        self.ax.set_yscale('log')
 #       self.ax.set_xscale('log')
        self.ax.set_yscale('linear')
        self.ax.set_xscale('linear')
        self.norm = plt.Normalize(1,4)
        self.cmap = plt.cm.RdYlGn
    
    def labels(self, labels):
        self.labels = labels
    
    def caption(self, x_coord, y_coord):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.annot = self.ax.annotate("", xy=(self.x_coord, self.y_coord), xytext=(20,20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        
    def scatter(self, x_points, y_points, c, marker=".", s=32):
        self.x_points = x_points
        self.y_points = y_points
        self.c = c
        self.marker = marker
        self.s = s
        self.sc = self.ax.scatter(self.x_points, self.y_points, c=self.c, marker=self.marker, s=self.s)
        self.cbar = self.fig.colorbar(self.sc, ticks=self.c)
        self.cbar.ax.set_yticklabels(self.c, size=14, fontname="Comic Sans MS")
        self.cbar.set_label('Gas\n(Arbitrary Units)', size=18)
    
    def update_annot(self, ind):
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = "{}".format([int(self.labels[n]) for n in ind["ind"]])
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor(self.cmap(self.norm(self.c[ind["ind"][0]])))
        self.annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
    
    def show(self):
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.show()
    
    def savefig(self, path, bbox_inches="tight", dpi=240):
        self.fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)
    
    def close(self):
        plt.close(self.fig)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

class Chart2D:
    def __init__(self, name):
        self.name = name
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_yscale('linear')
        self.ax.set_xscale('linear')
        self.lines = []
        #self.labels = []


    def caption(self, x_coord, y_coord):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.annot = self.ax.annotate("", xy=(self.x_coord, self.y_coord), xytext=(20,20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

    
    def update_annot(self, line, ind):
        sc = line.sc
        labels = line.labels
        pos = sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = "{}".format([(labels[n]) for n in ind["ind"]][0])
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)
        
        
    
    def scatter(self, x_points, y_points, name, labels, marker=".", s=32):
        # Make a Scatter2D object to be plotted later

        self.lines.append(Scatter2D(name,
                                    x_points,
                                    y_points,
                                    labels,
                                    marker,
                                    s))
        
        
        #self.labels.extend(labels)
        
        for i in range(len(x_points)):
            if labels[i]:
                self.caption(x_points[i], y_points[i])
                #self.labels.append(labels[i])
                
        
    
    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            
            for line in self.lines:
                cont, ind = line.sc.contains(event)
                if cont:
                    self.update_annot(line, ind)
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                else:
                    if vis:
                        self.annot.set_visible(False)
                        self.fig.canvas.draw_idle()
    
    
    def render(self):
        for line in self.lines:
            line.sc = self.ax.scatter(line.x_points,
                                      line.y_points,
                                      label=line.name)
            
            line.sc.descriptions = line.labels
            
    
    
    def show(self):
        self.render()
        self.ax.set_title(self.name)
        self.ax.legend()
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.show()
    
    def savefig(self, path, bbox_inches="tight", dpi=240):
        self.fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)
    
    def close(self):
        plt.close(self.fig)
        
        
        
        
        
        
        
        
class Scatter2D:
    def __init__(self,
                 name,
                 x_points,
                 y_points,
                 labels,
                 marker,
                 s):
        self.name = name
        self.x_points = x_points
        self.y_points = y_points
        self.labels = labels
        self.marker = marker
        self.s = s
        
        











































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

class Chart2DOld:
    def __init__(self, name):
        self.name = name
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_yscale('linear')
        self.ax.set_xscale('linear')
        self.cmap = cm.get_cmap("Spectral")
        self.scatterDict = {}
        self.labels = []


    def caption(self, x_coord, y_coord, label):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.labels.append(label)
        self.annot = self.ax.annotate("", xy=(self.x_coord, self.y_coord), xytext=(20,20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
       
    def scatter(self, x_points, y_points, name, labels, marker=".", s=32):
        # Make a Scatter2D object to be plotted later
        """
        input(f"name = {name}")
        input(f"xpoints = {x_points}")
        input(f"points = {y_points}")
        print("\n\n")
        """
        self.scatterDict[name] = Scatter2D(name,
                                            x_points,
                                            y_points,
                                            labels,
                                            marker,
                                            s)
            
        for i in range(len(x_points)):
            if labels[i]:
                self.caption(x_points[i], y_points[i], labels[i])
                
        
        
                
        #self.sc = self.ax.scatter(x_points, y_points, marker=self.marker, s=self.s)
        #self.ax.scatter(x_points, y_points, marker=marker, s=s)
    
    def update_annot(self, ind):
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        #text = "{}".format([f"{self.labels[n]}" for n in ind["ind"]])
        #text = "{}".format([self.labels[n] for n in ind["ind"]])
        text = [str(self.labels[n]) for n in ind["ind"]]
        self.annot.set_text(text)
        
       
        #self.annot.get_bbox_patch().set_facecolor(self.cmap(self.norm(self.c[ind["ind"][0]])))
        #self.annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
    
    
    def render(self):
        self.x_points = []
        self.y_points = []
        self.c = []
        self.ticklabels = ['']
        
        n = len(self.scatterDict)
        #self.norm = colors.Normalize(0, n)
        bounds = np.linspace(0, n, n+1)
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        i = 0
        
        for name in self.scatterDict:
            label = "\n\n" + self.scatterDict[name].name
            #self.ticklabels.append(self.scatterDict[name].name)
            self.ticklabels.append(label)
            self.x_points.extend(self.scatterDict[name].x_points)
            self.y_points.extend(self.scatterDict[name].y_points)
            
            
            for x in (self.scatterDict[name].x_points):
                self.c.append(i)
            i += 1
            
        
            
        



        self.sc = self.ax.scatter(self.x_points, self.y_points, c=self.cmap(self.norm(self.c)))
                                  #c=self.c)
#                                  c=self.c)
    
    
        self.cbar = self.fig.colorbar(cm.ScalarMappable(norm=self.norm, cmap=self.cmap), ax=self.ax)
        self.cbar.ax.set_yticklabels(self.ticklabels)
        for tick in self.cbar.ax.get_yticklabels():
            tick.set_verticalalignment("top")
            #tick.set_verticalalignment("bottom")
            #tick.set_verticalalignment("center")
            #tick.set_verticalalignment("baseline")
            #tick.set_verticalalignment("center_baseline")
    
    
    
    def show(self):
        self.render()
        self.ax.set_title(self.name)
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.show()
    
    def savefig(self, path, bbox_inches="tight", dpi=240):
        self.fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)
    
    def close(self):
        plt.close(self.fig)
        
        
        
        