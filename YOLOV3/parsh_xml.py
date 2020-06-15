from xml.dom.minidom import parse
dom=parse("./data/1.xml")
root=dom.documentElement#得到根节点
img_name=root.getElementsByTagName("path")[0].childNodes[0].data#得到名称
#提取图片尺寸
img_size=root.getElementsByTagName("size")[0]
img_w=img_size.getElementsByTagName("width")[0].childNodes[0].data
img_h=img_size.getElementsByTagName("height")[0].childNodes[0].data
img_c=img_size.getElementsByTagName("depth")[0].childNodes[0].data
#循环得到相对应的坐标值
objects=root.getElementsByTagName("item")
for box in objects:
    cls_name=box.getElementsByTagName("name")[0].childNodes[0].data
    x1=int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
    y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
    x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
    y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
    print(cls_name,x1,y1,x2,y2)


