import xml.etree.cElementTree as ET
import os

def write_xml(obj_infos, width, height, obj_info, classname, outputpath, filename):
    annotation = ET.Element('annotation')
    
    ET.SubElement(annotation, 'folder').text = 'tobedefined'
    ET.SubElement(annotation, 'filename').text = filename + '.png'
    ET.SubElement(annotation, 'path').text = os.path.join(outputpath, filename + '.png')
    
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    
    ET.SubElement(annotation, 'segmented').text = '0'
    
    for obj_info in obj_infos:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = str(obj_info['id'])
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(obj_info['bb'][0])
        ET.SubElement(bbox, 'ymin').text = str(obj_info['bb'][1])
        ET.SubElement(bbox, 'xmax').text = str(obj_info['bb'][2])
        ET.SubElement(bbox, 'ymax').text = str(obj_info['bb'][3])
    
    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(outputpath,filename + '.xml'))