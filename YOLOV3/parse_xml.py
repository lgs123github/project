from xml.dom import minidom

xml_path="./1.xml"
dom=minidom.parse(xml_path)#<xml.dom.minidom.Document object at 0x0000016D855A1948>
root=dom.documentElement#<DOM Element: doc at 0x187b9402f48>根节点

#先获取该元素节点，再获取子文本节点，最后通过“data”属性获取文本内容
img_name=root.getElementsByTagName("path")[0].childNodes[0].data

#获取图片长宽高
img_size=root.getElementsByTagName("size")[0]
img_w=img_size.getElementsByTagName("width")[0].childNodes[0].data
img_h=img_size.getElementsByTagName("height")[0].childNodes[0].data
img_c=img_size.getElementsByTagName("depth")[0].childNodes[0].data


print(img_c)









