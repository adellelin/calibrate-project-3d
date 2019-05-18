
import xml.etree.ElementTree as xmltree
import xml.dom.minidom as minidom
import numpy as np
import lxml.etree as etree

def check_numImages(root):
    images = root.iter('image')
    print("number of images", sum(1 for _ in images))

def remove_unlabeled(root):
    # find the images which haven't been tagged and remove from list, write to new file
    images = root.iter('image')
    for image in images:
        # if image is not labeled it will not have any children
        numChildren = len(image.getchildren())
        if numChildren == 0:
            print("attribute" ,image.attrib, root[2].tag)
            root[2].remove(image)

def get_points(root):
    '''
    read all the points from xml file for all faces
    creates an array with number of images and 68 points within
    eg if there are 5 images , the array will have the first shape index being number of images
    '''
    images = root.iter('image')

    pts_in_allimages = []

    for image in images:
        # if image is not labeled it will not have any children
        numChildren = len(image.getchildren())
        if numChildren == 0:
            pass
        else:
            pts = []  # pts in an image
            #prints the first point
            for i in range(68):
                #print("points", image.getchildren()[0].getchildren()[i].attrib)
                pts_dict = image.getchildren()[0].getchildren()[i].attrib
                #print(pts_dict['x'], pts_dict['y'])
                pts.append([int(pts_dict['x']), int(pts_dict['y'])])
            print(np.array(pts).shape)
            pts_in_allimages.append(pts)
    print(np.array(pts_in_allimages).shape)
    return np.array(pts_in_allimages)
            #pass

def get_imagesinlist(imageIter):
    imageArray = []
    for img in imageIter:
        #print("images",img.attrib, img.items()[0][1]) #.split('.',-1))
        imageArray.append(img.items()[0][1].split('/')[-1])
    #print("image array", imageArray)
    return imageArray

def get_imageIndeces(imageIter):
    '''
    if the images are in order, this will get the last 2 digits of the images to create
    an array of indeces
    '''
    imageArray = []
    for img in imageIter:
        print("images",img.attrib, img.items()[0][1]) #.split('.',-1))
        for k, v in img.items():
            image_no = v.split('.')[-2]
        imageArray.append(int(image_no))
    print("list of image count", sorted(imageArray))

def create_xml(): #points, imageStr, bb_ul, bb_lr):
    # create a new empty xml file
    root = xmltree.Element("dataset")
    augment = xmltree.Element("augmentations")
    images = xmltree.Element("images")
    root.append(augment)
    root.append(images)
    return root

def append_xml(root, points, imageStr, bb_ul, bb_lr):
    '''
    :param root: the xmlroot
    :param points: array of facial feature points (68,2)
    :param imageStr: name of image file
    :param bb_ul: top left bounding box point
    :param bb_lr: lower right bounding box point
    :return:
    '''

    images_element = root.find('images')
    image = xmltree.Element("image", file=imageStr)

    box_dict = get_bbox_dict(bb_ul, bb_lr)
    box = xmltree.SubElement(image, "box", box_dict)

    box = add_points(box, points)
    images_element.append(image)

    return root

def append_xml_augment(root, aug_type):
    augment_element = root.find('augmentations')
    augmentations = xmltree.Element('augment', type=aug_type)
    augment_element.append(augmentations)
    return root

def read_xml(xml_path):
    tree = xmltree.parse(xml_path)
    root = tree.getroot()
    # for elem in e[2][2][0].iter('part'):
    #     print(elem.attrib)
    # for elem in e.iter('image'):
    #     print(elem.attrib)
    check_numImages(root)
    pts = get_points(root)
    #remove_unlabeled(root)

    # find all the tags call 'image'
    imageIter = root.iter('image')
    images = get_imagesinlist(imageIter)
    #print("XML", images)
    return images, pts

#def read_xml_points(xml_path):


def get_bbox_dict(bb_ul, bb_lr):
    box_dict = {}
    box_dict["top"] = str(int(bb_ul[0]))
    box_dict["left"] = str(int(bb_ul[1]))
    box_dict["width"] = str(int(bb_lr[1] - bb_ul[1]))
    box_dict["height"] = str(int(bb_lr[0] - bb_ul[0]))
    return box_dict

def add_points(box, points):
    #points = points.reshape(3, 2)

    for i in range(points.shape[0]):
        pdict = {}
        pdict["name"] = format(i, '02')#str(i)
        pdict["x"] = str(int(points[i][0]))
        pdict["y"] = str(int(points[i][1]))
        #print("point dict", pdict)
        part = xmltree.SubElement(box, "part", pdict)
    return box

if __name__ == '__main__':
    """
    tree = xml.parse('/Users/adelleli/Documents/Git/featureseg/kubomom_0403_1437_imglab.xml')
    root = tree.getroot()
    # for elem in e[2][2][0].iter('part'):
    #     print(elem.attrib)
    # for elem in e.iter('image'):
    #     print(elem.attrib)
    check_numImages(root)
    get_points(root)
    remove_unlabeled(root)

    # find all the tags call 'image'
    imageIter = root.iter('image')
    get_imagesinlist(imageIter)

    #tree.write('/Users/adelleli/Documents/Git/featureseg/kubomom_0403_1437_imglab_reduced3.xml')
    check_numImages(root)
    """

    filename = "/Users/adelleli/Desktop/LAIKAOUTPUT/kubomom_train.xml"

    image_name='0600.0020.final.dshort.0001.0003.png'
    #bdict = {'top':'475', 'left':'1054', 'width':'523', 'height':'580'}
    bb_ul = [587, 1270]
    bb_lr = [889, 1567]
    points=np.array([[[1320, 673]],[[1334, 755]],[[1352, 820]]])
    points = points.reshape(3, 2)
    print(points.shape)

    root = create_xml() #points, image_name, bb_ul, bb_lr)

    image_name='0600.0020.final.dshort.0001.0004.png'
    bb_ul = [587, 1270]
    bb_lr = [889, 1567]
    points=np.array([[[1320, 673]],[[1334, 755]],[[1352, 820]]])
    points = points.reshape(3, 2)

    newroot = append_xml(root, points, image_name, bb_ul, bb_lr)
    image_name='0600.0020.final.dshort.0001.0005.png'
    bb_ul = [587, 1270]
    bb_lr = [889, 1567]
    points=np.array([[[1320, 673]],[[1334, 755]],[[1352, 820]]])
    points = points.reshape(3, 2)
    newnewroot = append_xml(root, points, image_name, bb_ul, bb_lr)

    # pdict = {'name':'0', 'x':'23', 'y':'100'}
    # pdict1 = {'name':'1', 'x':'23', 'y':'100'}
    # pdict2 = {'name':'2', 'x':'23', 'y':'100'}
    # part = xml.SubElement(box, "part", pdict)
    # part = xml.SubElement(box, "part", pdict1)
    # part = xml.SubElement(box, "part", pdict2)

    rough_string = xmltree.tostring(newnewroot, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    newtree = reparsed.toprettyxml(indent="\t")
    print(newtree)

    with open(filename,"w") as fh:
        fh.write(newtree)





