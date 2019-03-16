# How To Detect and Extract Faces from an Image using OpenCV and Python 

## Introduction

Images make up a large amount of the data that gets generated each day, which makes the ability to process these images important. One method of processing images is via *face detection*. Face detection is a branch of image processing that uses machine learning to detect faces in images.

[Haar Cascade](https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html) is an object detection method used to locate an object of interest in images. The algorithm is trained on a large number of positive and negative samples, where positive samples are images that contain the object of interest. Negative samples are images that may contain anything but the desired object. Once the classifier has been trained, it can locate the object of interest in any new images.

In this tutorial, you will use a pre-trained [HaaR Cascade](https://github.com/opencv/opencv/tree/master/data/haarcascades) from [OpenCV](https://opencv.org/) and [Python](https://www.python.org/) to detect and extract faces from an image. OpenCV is an open source programming library that is used to process images.


<!-- TODO: There is a lot of great information in this tutorial, but I think you could shorten things up by keeping the introduction focused on the goal of this tutorial. For example, I suggest using something like this:

	Images make up a large amount of the data that gets generated each day, which makes the ability to process these images important. One method of processing images is via *face detection*. Face detection is a branch of image processing that uses machine learning to detect faces in images.

	[Haar Cascade](https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html) is an object detection method used to locate an object of interest in images. The algorithm is trained on a large number of positive and negative samples, where positive samples are images that contain the object of interest. Negative samples are images that may contain anything but the desired object. Once the classifier has been trained, it can locate the object of interest in any new images.

	In this tutorial, you will use a pre-trained [HaaR Cascade](https://github.com/opencv/opencv/tree/master/data/haarcascades) from [OpenCV](https://opencv.org/) and [Python](https://www.python.org/) to detect and extract faces from an image. OpenCV is an open source programming library that is used to process images.

AS : Thanks for the recast. I have used it to replace the original content. bdw, anyway we can include the trivia here ? :)

	-->

## Prerequisites


* A [local Python 3 development environment](https://www.digitalocean.com/community/tutorial_series/how-to-install-and-set-up-a-local-programming-environment-for-python-3), including [pip](https://pypi.org/project/pip/), a tool for installing Python packages, and [venv](https://docs.python.org/3/library/venv.html), for creating virtual environments.

<!-- * HaaR Cascade File for the object that you want to extract. Pre-trained cascades for common objects such as face, hand, eyes etc are available publicly on OpenCV's [Github Repository](https://github.com/opencv/opencv/tree/master/data/haarcascades). Please download the suitable XML file from there. In this example we are going to use `haarcascade_frontalface_default.xml`.
TODO: Since you tell the reader how to download this file in a later step, can you remove this prerequisite? 

AS : Done

-->

## Step 1 — Configuring the Local Environment

Before you begin writing your code, you will first create a workspace to hold the code and install a few dependencies.
<!-- NOTE: In my suggested recast, I realized that I wrote "In this step, you will create a script that...". Since the reader doesn't actually write the script in this step, I recast this to remove that first sentence. Sorry about that! -->

Create a directory for the project with the `mkdir` command:

```command
mkdir face_scrapper
```

Change into the newly created directory:

```command
cd face_scrapper
```

Next, create a virtual environment for this project. Virtual environments isolate different projects so that differing dependencies won't cause any disruptions.
<!-- TODO: I really like this explanation that you've added. Can you align it more closely to our style by removing the usage of "we" and by using proper casing on words like "environments" and "Python"?
	
	Next, create a virtual environment for this project. Virtual environments isolate different projects so that differing dependencies won't cause any disruptions.  
AS : Done
	
-->

<!--
```command
pip install virtualenv
```
-->
<!-- TODO: The Python tutorial listed in the prerequisites covers installing virtualenv, so you don't need to include it here. You can have the reader create the new virtual environment without asking them to install it. Can you please update that? 
AS : Done
-->

As part of the pre-requisite steps we had already installed virtual environment. We will now create a virtual environment named `face_scrapper` to use with this project: 

```command
python3.6 -m venv <^>face_scrapper<^>
```

Activate the isolated environment:

```command
source <^>face_scrapper<^>/bin/activate
```

You will now see that your prompt is prefixed with the name of your virtual environment:

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
```
  
<!-- NOTE: I added a custom prefix to the steps executed from the command line. This will show the reader that the commands should be ran from their virtual environment.-->

Now that you've activated your virtual environment, you will use `nano` or your favorite text editor to create a `requirements.txt` file. This file will install the necessary Python dependencies:

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
nano requirements.txt
```
<!-- NOTE: I recast this to include a command for creating the requirements.txt file. Does that look OK to you? 
AS : Absolutely.
-->

Next, you need to install add three dependencies to complete this tutorial:

* `numpy`<!-- TODO: Should `npmpy` be changed to `numpy`? If so, can you please update it?  AS : Done.-->: [numpy](https://en.wikipedia.org/wiki/NumPy) is a Python library that adds support for large, multi-dimensional arrays. It also includes a large collection of mathematical functions to operate on the arrays.
* `opencv-utils`: This is the extended library for OpenCV that includes helper functions. <!-- TODO: Can you provide an example of an included helper function that this tutorial will use? -->
* `opencv-python`: This is the core OpenCV module that Python uses.

Add the following dependencies to the file:

```requirements.txt
[label requirements.txt]
numpy 
opencv-utils
opencv-python
```
<!-- TODO: Similar to above, should `npmpy` be changed to `numpy`? If so, please update it. 

AS : Done
-->

Save and close the file. Install the dependencies by passing the `requirements.txt` file to the Python package manager, `pip`. The `-r` flag specifies the location of `requirements.txt` file.

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
pip install -r requirements.txt
```

Finally, download the [HaaR Cascade file](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) for face detection from the OpenCV GitHub repository in your present directory. You will use this file shortly in your code.

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

In this step, you started with setting up a virtual environment for your project, installed the necessary dependencies, and downloaded the Haar-Cascade classifier that your code will use to detect faces. You're now ready to start writing the code to detect faces from an input image in next step.

<!-- TODO: Can you add a summary for this step? You could use something like this:

	In this step, you set up a virtual environment for your project, installed the necessary dependencies, and downloaded the Haar-Cascade classifier that your code will use to detect faces. You're now ready to start writing the code for your project. 

AS : Done
-->

## Step 2 — Writing and Running the Face Detector Script

In this section, you will write code that will take an input image and return two things:
<!-- TODO: Can you recast this to align more closely with our tone? The reader won't only be looking at code, but they will be writing the code. Some light recasting will help the reader feel more confident when approaching this step. Here's an example:

	In this step, you will write code that will take an image as input and return two things:
	
	* The number of faces found in the input image.
	* A new image with a rectangular plot around each detected face. 

AS : Done
-->

* Number of faces found in the input image
* A new output image with an rectangle plot around each of the face's found in the input image.

Start by creating a new file to hold your code:

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
nano app.py
```

<!-- We shall start by first importing libraries required by our program. Open a new file `app.py` in your current directory and add following two lines to it. -->
<!-- TODO: Per our style guidelines, your tutorial should give the reader the exact command needed to accomplish a task. In this case, you will need to provide the exact command needed to create a new file. Here's an example:

	Start by creating a new file to hold your code:

	```command
	nano app.py
	```

	In this new file, start writing your code by first importing the necessary libraries. You will import two modules here: `cv2` and `sys`. The `cv2` module imports the `OpenCV` library into the program, and `sys` is imports common Python functions such as `argv` that your code will use.

AS : Done
-->
<!-- We are importing two modules here, `cv2` and `sys`. The `cv2` module is used to import `OpenCV` library into the program and `sys` is used to import common python functions such as `argv` etc. -->
In this new file, start writing your code by first importing the necessary libraries. You will import two modules here: `cv2` and `sys`. The `cv2` module imports the `OpenCV` library into the program, and `sys` is imports common Python functions such as `argv` that your code will use.

```python
[label app.py]
import cv2
import sys
```
<!-- TODO: The `command` formatting should only be used for commands ran from the CLI. If this content should be added to a file, please be sure to use the label formatting outlined at do.co/style. Please also be sure to add an explanation of why the reader will need to import these modules. 
AS : Done
-->
<!-- TODO: It doesn't appear that the formatting feedback was addressed. Please be sure to add a code block label that represents the file name per our style guidelines. You should add this label on any code blocks that ask the reader to add code to a file. Here's the relevant link to the style guide: http://do.co/style#code-block-labels 
AS : Done
-->

<!-- Next we shall read the input image and convert it to a gray-scale. We do this using OpenCV's built in read(`imread`) and convert(`cvtColor`) functions. Please note that we shall be using gray scale version of the image here for processing as single channel( here would yield better results. This [stackoverflow thread](https://stackoverflow.com/questions/12752168/why-we-should-use-gray-scale-for-image-processing) gives an good insight into reasons behind it. 
AS : Done
-->
<!-- TODO: Thanks for adding in this explanation. Can you recast this paragraph to remove the Stack Overflow link and provide some more context from that link? Per our style guidelines, authors shouldn't send readers offsite to gather information that could be added to the article. This is because if this StackOverflow post were to be archived, that information will be gone and the reader will have no idea what was talked about. What do you think of something like this?

	A common practice in image processing is to first convert the input image to gray-scale. This is because detecting luminance, as opposed to color, will generally yield better results in object detection. Add the following code to take an input image as an argument and convert it to gray-scale: 
	
AS : Done	
-->

A common practice in image processing is to first convert the input image to gray-scale. This is because detecting luminance, as opposed to color, will generally yield better results in object detection. Add the following code to take an input image as an argument and convert it to gray-scale
```python
[label app.py]
...
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

The `.imread()` function takes the input image, which is passed as an argument to the script, and converts it to an OpenCV object. Next, OpenCV's `.cvtColor()` function converts the input image object to a gray scale object.

Now that you've added the code to load an image, you will add the code that detects faces in the specified image:

```python
[label app.py]
...
faceCascade = cv2.CascadeClassifier(cascPath)
faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
) 

print("Found {0} Faces!".format(len(faces)))

```

This code will first create a `.cascadeClassifier()` model. This will load the HaaR cascade file in to Python as an object called `faceCascade` so that it can be used by your code. 

Next, the code applies OpenCV's `.detectMultiScale()` method on the `faceCascade` object. This generates a _list of rectangles_ for all of the detected faces in the image. Here is a summary of the parameters used:

* `gray` - This refers to the OpenCV's gray scale image object that we loaded earlier. 
* `scaleFactor` - This parameter defines the scale at which the input image will be "shrunken down" for each pass. After each pass, the input image is scaled up/down by this factor. In other words, this parameter is also what lets you configure the speed and accuracy of the detection. Off course, this process is stopped once a threshold limit is reached, defined by `maxSize` and `minSize`. 
* `minNeighbors` - This parameter allows you to keep false positives out of final detection. HaaR based detection models work using sliding window approach. Under this approach, at the start of the detection a sized window is defined using `minSize` and `maxSize` parameters. That window is then slided across the image to detect for object. After each pass the window is resize and detection runs again. By default, the system is prone to detect many false positives. This parameter keeps those false positives out by imposing a condition that any detection needs to have at least this many neighbors. The idea behind this is that any true detection will have more detections located in that area. 
* `minSize` - This allows you to define minimum possible object size. Objects smaller than that are ignored. This parameter determine how small size you want to detect (in pixels).

Before moving forward, lets investigate this _list of rectangles_ that got returned from  `detectMultiScale` method. This list actually is a list of `pixel locations`, in the form of `Rect(x,y,w,h)`, for all the objects that were detected in the input image based on the cascade provided. We then used those `pixel locations` to create a bounding box around all the detected faces in the image.

<!-- TODO: It looks like this explanation came from StackOverflow. For copyright reasons, we can't actually reprint content that's already been published somewhere else. While it's OK to have published articles with similar topics elsewhere, make sure that all of the content of your DO article — including the writing — is unique. We publish our articles under an Attribution-NonCommercial-ShareAlike license (https://creativecommons.org/licenses/by-nc-sa/4.0/), so you can repost your DigitalOcean article with attribution on non-commercial sites after it's published here.

Additionally, can you tell the reader what `30` represents? 30 pixels? 30 centimeters? This information will help them understand what adjusting those parameters will do. 

AS : done

-->
    
After generating a list of rectangles, the faces are counted using `len` function. The number of found faces are then returned as output when the script is used.

Next, We use OpenCV's `.rectangle()` method to draw a rectangle around the detected faces.

```python  
[label app.py]
...
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

```
<!-- TODO: Can you explain in a little more detail what exactly this loop is doing? Try telling the reader what the `(image, (x, y), (x+w, y+h), (0, 255, 0), 2)` portion of code is doing.

AS : Done
-->
We use an for loop to iterate through the list of pixels returned from `faceCascade.detectMultiScale` method earlier for each of the detected object. The `rectangle` method typically takes four arguments

* `image` is the handle to input image which will be used by `rectangle` method to draw on.
* `(x,y), (x+w, y+h)` are the four pixel locations for the detected object. `rectangle` would use these to locate and draw an rectangle around the detected object in the input image.
* `(o, 255, 0)` is the color of the shape. For BGR, this argument gets passed as a tuple, eg: (255,0,0) for blue. We are using green in this case.
* `2` is the thickness of the line. 

Now that the rectangles are drawn, use OpenCV's `.imwrite()` method to write the new image to your local filesystem as `faces_detected.jpg`. This method will return `true` if the write was successful and `false` if it wasn't able to write the new image.

```python
...
status = cv2.imwrite('faces_detected.jpg', image)
```

Finally, add this code to print the return status (`true` or `false`) of the `imwrite` function to console. This will let you know if the write was successful after running the script.

```python
...
print ("Image faces_detected.jpg written to filesystem: ",status)
```

The completed file will look like this:

```python
[label app.py]
import cv2
import sys

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cascPath)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

status = cv2.imwrite('faces_detected.jpg', image)
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
```

Once you've verified that everything is entered correctly, save and close the file.

<$>[note]
**Note:** This code was sourced from the publicly available [OpenCV documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).
<$>

<!-- NOTE: I updated this disclaimer to use note formatting.

AS : Thanks :)
-->

Your code is complete and you are ready to run the script.

## Step 3 — Running the Script

In this step, you will use an image to test your script. When you find an image you'd like to use to test, save it in the same directory as your `app.py` script. This tutorial will use the following image: 

<!-- TODO: I have some concerns with using The Beatles photo due to copyright issues. Do you think that we can use a public domain image, such as this one from Pexel? https://i.imgur.com/TFI6HKc.jpg 

AS : I undersatnd. Let me use the new image. I am also going to update the flow of the article a little by showing the user input image first and output image later, instead of showing them side by side later as were doing it before. Thoughts ?

-->

Here is the input image that we are using with this tutorial.

![Input Image](https://i.imgur.com/TFI6HKc.jpg)

If you would like to test with the same image, use the following command to download it:

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
curl -O https://i.imgur.com/TFI6HKc.jpg
```
<!-- TODO: I added a step here to show the reader how to download an image with curl. If you are OK with using the above image, we will upload it to our server and add a custom link for it here. 

AS : Sounds right.

-->

Once you have an image to test the script, run the script and provide the image path as an argument:

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
python app.py <^>path/to/input_image<^>
```

Once the script finishes running, you will receive output like this:

```
[label Output]
[INFO] Found 4 Faces!
[INFO] Image faces_detected.jpg written to filesystem:  <^>True<^>
```

The `true` output tells you that the updated image was successfully written to the filesystem. Open the image on your local machine to see the changes on the new file:

![Output Image](https://i.imgur.com/4Np7sW8.jpg)

<!-- Image on the left is the input image which we downloaded earlier while the image on right is the output image that gets written to disk after the execution of the script. One curious thing that might be noticed between before and after images is that how the white background in the original image changes to actual (where it seems like Beatles are standing in backstage). This happened because most probably the input image had the background padded with white for better contrast and converting it to gray scale and back to color lost that additional padded information in the output image. -->
<!-- TODO: Can you explain to the reader where the image with the white background came from? The image on the left is not the true input image and it isn't too clear how or where you found it. If the picture on the left isn't the true input, please remove this explanation and update the image to only show the "after" result. 
AS : Done
-->

Shown above is  the output image that gets written to disk after the execution of the script. Our scipt detected four faces in the input image and drew rectangles to mark them. We shall be be using these rectangles in later part of the tutorial to extract them off the image.
## Step 4 — Extracting Faces and Saving them Locally (Optional)

<!-- TODO: I think that this explanation would work really well when explaining the code in the previous step to the reader. Can you move the `Rect(x,y,w,h)` explanation where it is first introduced? 
AS : Imho, this explanation works well here as it would provide context to the user which might be useful in following section. Your thoughts ?
-->
<!-- TODO: You are correct that is does provide context here, but the previous step doesn't provide much context around the list of rectangles. This means that the reader doesn't understand how the list of rectangles works until they get to this final step, which may leave them feeling confused until they get to this step. If you introduce these concepts when the reader first uses them, you can then elaborate further here on how these concepts are used. 

AS : I see your point. Have moved the explaination to preceeding section and added an statement in next para to maintain continuty.
-->
In the last step, you wrote code to use OpenCV and a HaaR cascade to detect and draw rectangles around faces in an image. In this section, you will modify your code to extract the detected faces from the image into their own files. Remember the _list of rectangles_ that got returned from `.detectMultiScale` method earlier. You will use that list to extract faces from the input image.

Start by reopening the `app.py` file with your text editor:

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
nano app.py
```

Next, add the highlighted lines under the `cv2.rectangle` line:" 

```python
[label app.py]
...
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    <^>roi_color = image[y:y + h, x:x + w]<^> 
    <^>print("[INFO] Object found. Saving locally.")<^> 
    <^>cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)<^> 
...
```
<!-- TODO: Reminder to use label formatting in this first code block. 
AS : Done
-->
<!-- 
Let's go through these lines one by one.

```python
[label app.py]
...
roi_color = image[y:y + h, x:x + w]
...
```
-->
<!-- TODO: After reviewing the updated draft, I think that your following explanations are clear enough that the reader will know which lines you are referencing. You can remove the individual lines and just use something like this:

	The `roi_color` object plots the pixel locations from the `faces` list on the original input image. The `x`, `y`, `h`, and `w` variables are the pixel locations for each of the objects detected from `faceCascade.detectMultiScale` method. The code then prints output stating that an object was found and will be saved locally.

	Once that is done, the code saves the plot as a new image using the `cv2.imwrite` method. It appends the width and height of the plot to the name of the image being written to. This will keep the name unique in case there are multiple faces detected.

	The updated `app.py` script will look like this:
	
AS : Done. Thanks :)	
	-->


<!-- The `roi_color` object plots the pixel locations from the `faces` list on the original input image. The `x`, `y`, `h`, and `w` variables are the pixel locations for each of the objects detected from `faceCascade.detectMultiScale` method.


```python
print("[INFO] Object found. Saving locally.")
```

Next, the code prints output stating that an object was found and will be saved locally.

```python
...
cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
```

This code will save the plot as a new image using the `cv2.imwrite` method. It appends the width and height of the plot to the name of the image being written to keep the name unique in case there are multiple faces detected.

The updated `app.py` script will look like this:-->

The `roi_color` object plots the pixel locations from the `faces` list on the original input image. The `x`, `y`, `h`, and `w` variables are the pixel locations for each of the objects detected from `faceCascade.detectMultiScale` method. The code then prints output stating that an object was found and will be saved locally.

Once that is done, the code saves the plot as a new image using the `cv2.imwrite` method. It appends the width and height of the plot to the name of the image being written to. This will keep the name unique in case there are multiple faces detected.

The updated `app.py` script will look like this:

```python
[label app.py]
import cv2
import sys

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cascPath)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces.".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    <^>roi_color = image[y:y + h, x:x + w]<^>
    <^>print("[INFO] Object found. Saving locally.")<^>
    <^>cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)<^>

status = cv2.imwrite('faces_detected.jpg', image)
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
```

To summarize, the added code uses the pixel locations to extract the faces from the image into a new file. Save and close the file. 

Now that you've updated the code, you are ready to run the script once more:

```custom_prefix((face_scrapper)\sSammys-MBP:~/face_scrapper\s#)
python app.py <^>path/to/image<^>
```

You will see the similar output once your script is done processing the image:

```
[label Console Output]
[INFO] Found 4 Faces.
[INFO] Object found. Saving locally.
[INFO] Object found. Saving locally.
[INFO] Object found. Saving locally.
[INFO] Object found. Saving locally.
[INFO] Image faces_detected.jpg written to file-system: <^>True<^>
```

Depending on how many faces are in your sample image, you may see more or less output.

Looking at the contents of the working directory after the execution of the script, you should see head shots of all faces found in the input image.

![Directory Listing ](https://i.imgur.com/GYS2Tnd.jpg)

You will now see head shots extracted from the input image collected in the working directory 

![Head shots](https://i.imgur.com/qxOEcmr.jpg)

In this step, you modified your script to extract the detected objects from the input image and save them locally. 

## Conclusion

In this tutorial, you wrote a script that uses OpenCV and Python to detect, count, and extract faces from an input image. You can update this script to detect different objects by using a different pre-trained Haar cascade from the OpenCV library, or you can learn how to [train your own](https://docs.opencv.org/3.3.0/dc/d88/tutorial_traincascade.html) Haar cascade.

<!--Machine learning in general and Image Processing in particular are changing the world as we see today. From Medical to manufacturing, BFSI to engineering, pretty much all modern fields now employ some kind of Image processing. The tutorial listed here barely begins to scratch the surface of possibilities of things that become possible with image processing. My humble hope is that this tutorial would ignite interest in a new user and act as an stepping stone for the experienced to take the next step in this field.

As shown above, with a few lines of code, we wrote an automated application which can detect, count and extract faces from an input image. A interested user can extend the application for many real world usage. Some that come to mind:

* Using a `full_body_haar_cascade.xml` available at OpenCV's Github repository, once can write a similar app to count the number of pedestrians in an input image. 
* Any automated cataloging system would benefit from an application of similar nature. Please note that the user would in that case need to train their own Haar Cascade model for that object.
* Automated polling system from images where the object to be detected is automatically detected using a application of similar nature. The list goes on.

As always, Happy Coding !!
-->
<!-- TODO: I appreciate the enthusiasm, but I think you can show the reader your enthusiasm by recasting this conclusion in a way that would align more closely with our style guidelines and focuses on what the reader accomplished in the tutorial. Can you use something like this?

	In this tutorial, you wrote a script that uses OpenCV and Python to detect, count, and extract faces from an input image. You can update this script to detect different objects by using a different pre-trained Haar cascade from the OpenCV library, or you can learn how to [train your own](https://docs.opencv.org/3.3.0/dc/d88/tutorial_traincascade.html) Haar cascade. 

AS : Done
	
-->
