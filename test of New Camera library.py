import v4l2 
import fcntl 
vd = open('/dev/video0', 'rw') 
cp = v4l2.v4l2_capability() 
fcntl.ioctl(vd, v4l2.VIDIOC_QUERYCAP, cp) 
#0 
cp.driver 
#'uvcvideo' 
cp.card 
#'USB 2.0 Camera' 