import cv2

args = dict(
    img_path=r'I:\GraduationThesis\SubstractNet\diff_resblocks\projectted\1/1.JPG',
    points_x=[],
    points_y=[],
    size=(1920, 1080),
    src_size=[0, 0, 0],

)
img = cv2.imread(args['img_path'])
args['src_size'][0], args['src_size'][1], channel = img.shape
img = cv2.resize(img, args['size'])

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(args['points_y']) < 5:
        xy = "%d,%d" % (x, y)
        args['points_x'].append(x)
        args['points_y'].append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x, y)
    else:
        cor_pt = [int((args['points_x'][0]/args['size'][0])*args['src_size'][0]),
                  int((args['points_y'][0]/args['size'][1])*args['src_size'][1])]
        rows = args['points_x'][1] - args['points_x'][0]
        cols = args['points_y'][3] - args['points_y'][0]
        rows = int((rows/args['size'][0])*args['src_size'][0])
        cols = int((cols/args['size'][1])*args['src_size'][1])
        print("point: ({},{}), rows: {}, cols: {}".format(cor_pt[0], cor_pt[1], rows, cols))
        exit(0)

def getClickPoint():
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__=='__main__':
    getClickPoint()
