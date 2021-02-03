# Live Image Mosaicking - lab exercise

Supporting files for an OpenCV Python based live (real-time) image mosaicking lab exercise used for teaching within the undergraduate Computer Science programme
at [Durham University](http://www.durham.ac.uk) (UK) by [Prof. Toby Breckon](https://breckon.org/toby/).

**This is not a complete real-time mosaicking code solution** _but instead supporting code files to allow you to build a real-time mosaicking solution._

![Python - PEP8](https://github.com/tobybreckon/mosaic-lab/workflows/Python%20-%20PEP8/badge.svg)

All tested with [OpenCV](http://www.opencv.org) 4.x and Python 3.x.

---

### Files:

- _skeleton.py_ - an outline file with camera/video interface and commented outline of the bits you need to complete
- _mosaic_support.py_ - a set of documented supporting functions

---

### How to download and run:

Download each file as needed or to download the entire repository and run each try:

```
git clone https://github.com/tobybreckon/mosaic-lab.git
cd mosaic-lab
python3 ./skeleton.py [optional video file]
```

Runs with a webcam connected or from a command line supplied video file of a format OpenCV supports on your system (otherwise edit the script to provide your own image source).

N.B. you may need to change the line near the top that specifies the camera device to use on some examples below - change "0" if you have one webcam, I have it set to "1" to skip my built-in laptop webcam and use the connected USB camera.

---

### Background:

Inspired and informed by the research work undertaken in:

[**Real-time Construction and Visualization of Drift-Free Video Mosaics from Unconstrained Camera Motion**](https://breckon.org/toby/publications/papers/breszcz15mosaic.pdf) (M. Breszcz, T.P. Breckon), In IET J. Engineering, IET, Volume 2015, No. 16, pp. 1-12, 2015 [[**demo**]](https://www.youtube.com/embed/videoseries?list=PLjKaMtzV6REx2fDm73bR99DM6f9nd2rbV) [[**pdf**]](https://breckon.org/toby/publications/papers/breszcz15mosaic.pdf) [[**doi**]](http://dx.doi.org/10.1049/joe.2015.0016)

---

If referencing these examples in your own work please use:
```
@Article{breszcz15mosaic,
  author = 	 {Breszcz, M. and Breckon, T.P.},
  title = 	 {Real-time Construction and Visualization of Drift-Free Video Mosaics from Unconstrained Camera Motion},
  journal = 	 {IET J. Engineering},
  year = 	 {2015},
  volume = 	 {2015},
  number = 	 {16},
  pages = 	 {1-12},
  month = 	 {August},
  publisher =    {IET},
  doi = 	 {10.1049/joe.2015.0016},
}
```

---

If you find any bugs raise an issue (or much better still submit a git pull request with a fix) - toby.breckon@durham.ac.uk

_"may the source be with you"_ - anon.
