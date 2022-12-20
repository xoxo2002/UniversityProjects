import cv2
import numpy as np 
from matplotlib import pyplot as plt

def main():   
    #read image 
    img = cv2.imread("008.png")
    plt.figure()  
    plt.imshow(img)
    
    #check image for tables and delete 
    img, max = remove_table(img)
   
    
    #initialise array to store individual paragraphs
    final = []
    
    #extract columns
    img = extract_column(img)
    no_of_para = 1
    
    #for each column check for tables and extract each paragraph
    for i in img:
        img, max = remove_table(i, 1)
        paragraph = extract_paragraph(img, max)
        final.append(paragraph)
    
    #display and save paragraphs into png files
    for i in range(len(final)):
        for j in range(len(final[i])):
            plt.figure()
            plt.imshow(final[i][j])
            plt.title("paragraph_0{}.png".format(no_of_para))
            cv2.imwrite("paragraph_0{}.png".format(no_of_para), final[i][j])
            no_of_para += 1
            
def remove_table(img, x = 0):

    binarizedImage = binarise_Image(img)
    height, width = binarizedImage.shape

    
    horizontal_px = np.sum(binarizedImage, axis=1) #horizontal bins
    paragraphs = np.where(horizontal_px > 0) #select coordinates of bins where there are words
    spaces = np.diff(paragraphs) # a list of values for the start and end of each lines 
    
    max = 0
    # distinguish between spacing of lines and spacing of paragraphs
    for i in range(spaces.size): 
        if spaces[0][i] > max:
            max = spaces[0][i]
    
    length = [] # find the length of continued vertical pixels
    s = [] #a list to record the start of continued pixels
    e = [] #a list to record the end of continued pixels
    n = 0
    
    #loop through all bins 
    for i in range(len(horizontal_px)):
        #find range where there are continuous presence of pixels along the vertical axis
        if horizontal_px[i] > 0 and n == 0: #start of pixel 
            n+=1
            s.append(i)
        elif horizontal_px[i] > 0 : #continued pixel
            n+=1
        elif (n > 0 and horizontal_px[i] == 0): #end of continued pixels and record the length of continued pixels
            length.append(n)
            n = 0
            e.append(i)
            
    for l in range(len(length)):
            #to find the table lines and turn area of table to white pixels
            if x == 0: #for more than 1 column
                if length[l] > max and (horizontal_px[s[l]+1] > (40/100 * width)): #if the length of the continued pixels is more than the max spacing and the horizontal length more than 20% of the width, it means the line of the table
                    img[s[l] - 20: e[l] + 20] = 255 # +-20 for any margin of error 
            else: #for after extracting column
                 if length[l] > max:
                     img[s[l] - 20: e[l] + 20] = 255

    return(img, max)
            
def extract_column(img):
    #create black background and white words for histogram projection
    binarizedImage = binarise_Image(img)

    
    # the bins for vertical pixels in histogram projection
    vertical_px = np.sum(binarizedImage, axis=0)
    words = (np.where(vertical_px > 0)) # coordinates where there are words (has pixels)
    spaces = np.where(np.diff(words) > 10) # see where there are large spaces between words (spaces between columns)
    spaces = spaces[1] # structure of the np.array
    print(spaces)
    
    #initialise array for start of column
    start = [] 
    end = [] 
    start.append(words[0][0])
    #from the differences arrange them into column starting and ending coordinates
    for i in spaces:
        end.append(words[0][i])
        start.append(words[0][i+1])
    end.append(words[0][-1])
    
    #extract the columns based on coordinates and store them in an array
    columns = []
    #print out all columns
    for i in range(len(start)):
        columns.append(img[:, start[i]-50: end[i]+50]) #+-50 is for the padding of left and right paddings of the columns
    
    return columns

def extract_paragraph(img, max):
    binarizedImage = binarise_Image(img)
    
    #bins for horizontal pixels
    horizontal_px = np.sum(binarizedImage, axis=1);
    lines = np.where(horizontal_px > 0)
    spaces = np.where(np.diff(lines) >= max - 10) #its another paragraph when the different between the line is >= max - 1
    spaces = spaces[1]
    
    #initialise an array for starting and the end of the row 
    start = []
    end = []
    #from spaces arrange the coordinates into starting and ending coordinates
    start.append(lines[0][0])
    for i in spaces:
        end.append(lines[0][i-1])
        start.append(lines[0][i+1])
    end.append(lines[0][-1])
    
    #use the coordinates to slice out copies paragraphs from the original image
    rowed_image = []
    for i in range(len(start)):
        rowed_image.append(img[start[i]-50: end[i]+50, :]) #+-50 for top and bottom paddings of extracted paragraphs
        
    return rowed_image

def binarise_Image(img):
    #turn image into gray scale
    grayImage= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    #threshold the image to get a binarised image
    _, binarizedImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
     
    #turn all words into 1 (white) and all spaces to 0 (black) for histogram projection
    binarizedImage[binarizedImage == 0] = 1
    binarizedImage[binarizedImage == 255] = 0
    
    return binarizedImage
        

if __name__ == "__main__":
    main()
