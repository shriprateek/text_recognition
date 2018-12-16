import argparse
import cv2
import os
from ctpn_boxes import ctpn_boxes
import pytesseract
from PIL import Image
import json

"""
Get cropped image

@param bbox: bounding box coordinates
@param img: image to crop
@rtype: list
@returns: cropped img
"""
def get_crop_img(bbox,img):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]
    img_crop = img[y_min:y_max,x_min:x_max]
    return img_crop

"""
Get Tesseract output

@param psm: page segmentation mode
       default =6: assumes uniform block of text
@param img: image to pass into tesseract
@rtype: str
@returns: output of tesseract
"""
def get_tesseract_output(img,psm = 6):
    filename = "out.png"
    cv2.imwrite(filename, img)
    
    '''
    -l eng: language is english
    -oem 1: OCR Engine Mode is LSTM only
    --tessdata-dir: directory where LSTM tesseract data is present
    '''
    
    config = (" -l eng --oem 1 --psm " + str(psm) +" --tessdata-dir tessdata")
    text = pytesseract.image_to_string(Image.open(filename),config=config)
    os.remove(filename)
    return text


"""
Get Tesseract output

@param bboxes: bounding boxes from CTPN
@param img: input image
@rtype: list
@returns: tesseract ouput of entire image
"""
def get_output_text(bboxes,img):
    output_text = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        
        img_crop = img[y_min:y_max,x_min:x_max]
        text = get_tesseract_output(img_crop)
        output_text.append(text)
    return output_text

"""
Get document type of the image
@param bboxes : bounding boxes by CTPN
@param img : input image
@returns: True if DL False if Voter_ID
"""
def get_doc_type(bboxes,img):
    img_crop = get_crop_img(bboxes[0],img)
    text = get_tesseract_output(img_crop)
    return 'Transport'in text

"""
Strip special characters in the name
@param text: input text
@returns: output text
"""
def strip_special_chars(text):
    out = ''
    out = out.join(e for e in text if (e.isalnum() or e == ' '))
    out = out.lstrip()
    return out

"""
Get name of the card holder

@param doc_type: 0 if Voter_ID, 1 if DL
@param output_text: Entire output text of tesseract
@returns: Name of the card holder

Heuristics:

Voter_ID: Search "Elector's Name" in output text and
          return output after ':'

DL:       Search "Name" in output text and return 
          text of next element in output text
"""
def get_name(doc_type,output_text):
    name = ""
    if (doc_type == 0): #Voter ID
        name_block = [x for x in output_text if 'Elector\'s Name' in x][0] #[0] for getting string directly
        name = name_block.split(': ')[1]
    else:
        name = output_text[output_text.index('Name') + 1]
    name = strip_special_chars(name)
    return name

"""
Get Father's/Husband's name of the card holder

@param doc_type: 0 if Voter_ID, 1 if DL
@param output_text: Entire output text of tesseract
@returns: Father's/Husband's name of the card holder

Heuristics:

Voter_ID: Search "Husband's Name" in output text and
          return output after ':'

DL:       Search "S/W/D" in output text and return 
          text of next element in output text
"""
def get_father_name(doc_type,output_text):
    name = ""
    if (doc_type == 0): #Voter ID
        name_block = [x for x in output_text if 'Husband\'s Name' in x][0] #[0] for getting string directly
        name = name_block.split(': ')[1]
    else:
        name = output_text[output_text.index('S/W/D') + 1]
    name = strip_special_chars(name)
    return name

"""
Get ID number of the card holder

@param doc_type: 0 if Voter_ID, 1 if DL
@param output_text: Entire output text of tesseract
@returns: ID number of the card holder

Heuristics:

Voter_ID: Search "IDENTITY CARD" in output text and
          return output after 'CARD'

DL:       Search "Licence No" in output text and return 
          output after ': '
"""
def get_id_num(doc_type,output_text):
    id_num = ""
    if doc_type == 0:
        id_block = [x for x in output_text if 'IDENTITY CARD' in x][0]
        id_num = id_block.split("CARD")[1].lstrip()
    else:
        id_block = [x for x in output_text if 'Licence No' in x][0]
        id_num = id_block.split(': ')[1]
    return id_num.upper()

"""
Get Date of Birth of the DL holder

@param output_text: Entire output text of tesseract
@returns: DoB of the DL holder

Heuristics:

Search "DOB" in output text and return output after ' '
"""
def get_dob(output_text):
    dob_block = [x for x in output_text if 'DOB' in x][0]
    dob = dob_block.split(" ")[1]
    return dob

"""
Get Blood Group of the DL holder

@param output_text: Entire output text of tesseract
@returns: BG of the DL holder

Heuristics:

Search "BG" in output text and return output after 'BG: '
Tesseract might recognize 'O' as '0'
"""
def get_bg(output_text):
    bg_block = [x for x in output_text if 'BG' in x][0]
    bg = bg_block.split("BG: ")[1]
    if bg == '0':
        bg = 'O'
    return bg

"""
Get Age of the Voter_ID card holder

@param output_text: Entire output text of tesseract
@returns: Age of the card holder

Heuristics:

Search "Age" in output text and return output after ':  '
"""
def get_age(output_text):
    age_block = [x for x in output_text if 'Age' in x][0]
    age = age_block.split(": ")[1]
    return age  

"""
Get Gender of the Voter_ID card holder

@param output_text: Entire output text of tesseract
@param boxes: CTPN Bounding boxes on input image
@param img: input image
@returns: Gender of the card holder

Heuristics:

Search "Sex" in output text and pass the cropped
image of the next bounding box in tesseract with
psm = 9 to treat as single word in a circle to get
the gender as Male or Female.
"""
def get_gender(output_text,boxes,img):
    gender_idx = output_text.index('Sex') + 1
    img_crop = get_crop_img(boxes[gender_idx],img)
    
    #psm = 9: Treat the image as a single word in a circle.
    gender = get_tesseract_output(img_crop,psm = 9)
    if 'M' in gender:
        gender = 'Male'
    elif 'F' in gender:
        gender = 'Female'
    else :
        gender = 'Not Recongized'
    return gender

"""
Get Address of the DL holder

@param output_text: Entire output text of tesseract
@param boxes: CTPN Bounding boxes on input image
@param img: input image
@returns: Address of the card holder

Heuristics:

Since Address keyword is not present in template
but location of address is below blood group 
and above Date of issue, so I formulated the bbox
coordinates of address from these two fields
and obtained the address by passing it into the tesseract.
"""
def get_address(boxes,img,output_text):
    bg_block = [x for x in output_text if 'BG' in x][0]
    bg_idx = output_text.index(bg_block)
    
    doi_block = [x for x in output_text if 'Date of Issue' in x][0]
    doi_idx = output_text.index(doi_block)
    
    '''
    xmin, ymin = lower left coordinates of 'blood group' bounding box
    xmax, ymax = top right coordinates of 'date of issue' bounding box
    '''
    
    xmin = boxes[bg_idx][0] 
    xmax = boxes[doi_idx][2]
    ymin = boxes[bg_idx][3]
    ymax = boxes[doi_idx][1]
    
    address_box = [xmin,ymin,xmax,ymax]
    img_crop = get_crop_img(address_box,img)

    address = get_tesseract_output(img_crop)
    address = address.replace('\n',' ')
    return address

if __name__ == '__main__':
    
    #Parser to get image path from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("image",type =str, help="enter the image path")
    args = parser.parse_args()
    image = args.image
    
    #Read image
    img = cv2.imread(image)
    
    #Obtain CTPN bounding boxes over the image
    bboxes = ctpn_boxes(img)
    output_text = get_output_text(bboxes,img)

    #Get document type
    doc_type = get_doc_type(bboxes,img)
    
    #Get ID card owner's name
    name = get_name(doc_type,output_text)
    
    #Get Father's/Husband's name
    father_name = get_father_name(doc_type,output_text)
    
    #Get ID card number
    id_num = get_id_num(doc_type,output_text)

    #Create dictionary to write the extracted entities into
    data = {}  
    if doc_type:
        #Get date of birth of the card holder
        dob = get_dob(output_text)
        
        #Get blood group of the card holder
        bg = get_bg(output_text)
        
        #Get address of the card holder
        address = get_address(bboxes,img,output_text)
        
        data['DL'] = []  
        data['DL'].append({
            'License Id Number': id_num,
            'Name': name,
            'Father/Husband Name': father_name,
            'DoB/YoB': dob,
            'Address': address,
            'Blood Group': bg
        })
    else:
        age = get_age(output_text)
        gender = get_gender(output_text,bboxes,img)
        
        data['VID'] = []  
        data['VID'].append({ 
            'Voter Id Number': id_num,
            'Name': name,
            'Gender':gender,
            'Father/Husband Name': father_name,
            'Age': age     
        })
            
    #Write information to json file in the same path as image
    json_filename = image.replace('.png','.json')
    with open(json_filename, 'w') as outfile:  
        json.dump(data, outfile,indent=4)
