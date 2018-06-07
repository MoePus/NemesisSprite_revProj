from struct import pack,unpack
from PIL import Image
import numpy as np
import os

unitId,unitSprType = (600081,"adv")
canvasSize = {
    "battle":480,
    "adv":960
}

fnameSprite = r"SpriteAnim/%s/%d_%s" % (unitSprType,unitId,unitSprType)
fnamePng = r"Image/%s/%d_%s.png" % (unitSprType,unitId,unitSprType)

spriteName = os.path.basename(fnameSprite)
unitId,unitSprType = spriteName.split("_")

img = Image.open(fnamePng)

index2DiscrpLut = {
    "battle":{
        "animpack":{
            0:"SA_PACK_UNIT_BATTLE"
        },
        "anim":{
            0:"SA_ANIM_UNIT_ATTACK",
            1:"SA_ANIM_UNIT_DAMAGE",
            2:"SA_ANIM_UNIT_NOSALLY",
            3:"SA_ANIM_UNIT_IDLE",
            4:"SA_ANIM_UNIT_WIN",
            5:"SA_ANIM_UNIT_SEARCH"
        }
    },
    "adv":{
        "animpack":{
            0:"SA_PACK_UNIT_ADV"
        },
        "anim":{
            0:"SA_ANIM_UNIT_IDLE",
            1:"SA_ANIM_UNIT_SMILE",
            2:"SA_ANIM_UNIT_ANGRY",
            3:"SA_ANIM_UNIT_WORRY"
        }
    },
}

def cropImage(img,LTRB):
    width,height = img.size
    return img.crop(LTRB * np.array([width,height,width,height])).convert("RGBA")

def scaleImage(img,scale):
    newSize = img.size*np.array(scale)
    if img.size[0] <0:
        img.transpose(Image.FLIP_LEFT_RIGHT)
    if img.size[1] <0:
        img.transpose(Image.FLIP_TOP_BOTTOM)

    newSize = np.abs(newSize)
    if newSize[0] < 1 or newSize[1] < 1:
        return Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    else:
        return img.resize(newSize.astype(int))


def GenRead4Bytes(type):
    def read4Bytes(fp,num = 0):
        try:
            num = int(num)
            _num = max(1, num)
            data = fp.read(4 * _num)
            data = unpack("%d%s"%(_num,type), data)
            if num == 0:
                return data[0]
            else:
                return data
        except Exception as e:
            print(hex(fp.tell()))
            raise e
    return read4Bytes

readInt = GenRead4Bytes("i")
readFloat = GenRead4Bytes("f")
readUInt = GenRead4Bytes("I")

def parseDivided(fp):
    compositeType = readInt(fp)
    maxTime = 0.0
    info = {
        "box":[],
        "translateX":[],
        "translateY":[],
        "Rotate":[],
        "ScaleX":[],
        "ScaleY":[],
        "alpha": [],
        "children":[],
        "layer":[]
    }
    def parseFrames(fp,infoCount,container,handler):
        maxTime = 0.0
        for i in range(infoCount):
            processed = handler(fp)
            container.append(processed)
            maxTime = max(maxTime,processed["time"])
        return maxTime

    def dividedHandler(fp):
        time = readFloat(fp)
        readInt(fp, 2)
        box = readFloat(fp, 4)
        LTRB = readFloat(fp, 4)
        return {
            "time":time,
            "value":LTRB,
            "extra":box
        }

    def translateHandler(fp):
        time = readFloat(fp)
        flag = readInt(fp, 2)
        unk = readFloat(fp, 3)
        translate = readFloat(fp)
        return {
            "time": time,
            "valid": flag[0],
            "value": translate
        }

    def unkHandler(fp):
        time = readFloat(fp)
        unk = readUInt(fp, 2)
        return {
            "time": time,
            "value":unk[1]
        }

    if True:
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp,infoCount,info["box"],dividedHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, info["translateX"], translateHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, info["translateY"], translateHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, info["Rotate"], translateHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, info["ScaleX"], translateHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, info["ScaleY"], translateHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, info["alpha"], translateHandler))

        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, info["layer"], unkHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, [], unkHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, [], unkHandler))
        infoCount = readInt(fp)
        maxTime = max(maxTime,parseFrames(fp, infoCount, [], unkHandler))

    infoCount = readInt(fp)
    if infoCount:
        for i in range(infoCount):
            fp.seek(fp.tell()+108)
    infoCount = readInt(fp)
    if infoCount:
        for i in range(infoCount):
            fp.seek(fp.tell() + 56)

    childCount = readInt(fp)
    for i in range(childCount):
        info["children"].append(readInt(fp))


    friendlyInfo = {}

    class interpolator:
        def __init__(self,args,default):
            self.default = np.array(default)
            times = set()
            for arg in args:
                times.update([x["time"] for x in arg])

            times = sorted(times)
            self.data = {}
            for time in times:
                argData = [None] * len(args)
                for i in range(len(args)):
                    srtTimeList = sorted(args[i],key=lambda x:abs(x["time"]-time))
                    if srtTimeList[0]["time"] <= 1e-4:
                        argData[i] = srtTimeList[0]["value"]
                    else:
                        argData[i] = default[i]

                self.data[time] = np.array(argData)

        def __getitem__(self, time):
            mytimes = sorted(self.data.keys())
            if len(mytimes) == 0:
                return self.default

            if time <= mytimes[0]:
                return self.data[mytimes[0]]
            elif time >= mytimes[-1]:
                return self.data[mytimes[-1]]

            nearTimes = sorted(mytimes,key=lambda x:abs(x-time))
            nearTimes = sorted(nearTimes[0:2])
            timeA = nearTimes[0];timeB = nearTimes[1];
            interval = timeB - timeA

            #return np.interp(time,[timeA,timeB],[self.data[timeA],self.data[timeB]])
            return self.data[timeA] * (timeB - time)/interval + \
                self.data[timeB] * (time - timeA) / interval

    if False:
        def vaildFilter(x):
            return x["valid"]
        info["translateX"] = list(filter(vaildFilter,info["translateX"]))
        info["translateY"] = list(filter(vaildFilter,info["translateY"]))
        info["ScaleX"] = list(filter(vaildFilter,info["ScaleX"]))
        info["ScaleY"] = list(filter(vaildFilter,info["ScaleY"]))
        info["Rotate"] = list(filter(vaildFilter,info["Rotate"]))

    friendlyInfo["box"] = interpolator([info["box"]], [[0,0,0,0]])
    friendlyInfo["translate"] = interpolator([info["translateX"], info["translateY"]], [0.0, 0.0])
    friendlyInfo["scale"] = interpolator([info["ScaleX"], info["ScaleY"]], [1.0, 1.0])
    friendlyInfo["rotate"] = interpolator([info["Rotate"]], [0.0])
    friendlyInfo["alpha"] = interpolator([info["alpha"]], [1.0])
    friendlyInfo["children"] = info["children"]

    extra = [{
        "time":x["time"],
        "value":[x["extra"][0]+x["extra"][2],x["extra"][1]+x["extra"][3]]
              } for x in info["box"]]
    friendlyInfo["extra"] = interpolator([extra], [[0, 0]])

    layer = [{
        "time": x["time"],
        "value": float(x["value"] - 0x80000000)
    } for x in info["layer"]]
    friendlyInfo["layer"] = interpolator([layer], [0.0])

    friendlyInfo["compositeType"] = compositeType
    return friendlyInfo,maxTime

def drawDivides(divides,time):
    canvas = Image.new('RGBA', (canvasSize[unitSprType], canvasSize[unitSprType]), (0, 0, 0, 0))
    center = np.array([canvasSize[unitSprType]/2, canvasSize[unitSprType]/2])

    class Matrix:
        def __init__(self,matrix = None):
            if matrix:
                self.data = matrix.data.copy()
            else:
                self.data = np.matrix([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]])

        def __translate(self,translate):
            self.data[3] = np.array([translate[0],translate[1],0,1]) * self.data

        def __rotateZ(self,theta):
            M_z = np.matrix([[np.cos(theta),np.sin(theta),0,0],
                            [-np.sin(theta),np.cos(theta),0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
            self.data = self.data * M_z

        def __scale(self,scale):
            self.data[0] = self.data[0] * scale[0]
            self.data[1] = self.data[1] * scale[1]

        def update(self,translate,scale,rotate):
            self.__translate(translate)
            self.__rotateZ(rotate)
            self.__scale(scale)

        def getXY(self):
            return np.squeeze(np.asarray(self.data[3]))[0:2]

        def translateR(self,translate):
            return np.squeeze(np.asarray(np.array([translate[0],translate[1],0,1]) * self.data))[0:2]


    croppeds = []
    def drawCropped(divided,translate,scale,rotate,alpha,matrix,index):
        translate = translate + divided["translate"][time]
        scale = scale * divided["scale"][time]
        rotate = rotate + divided["rotate"][time][0]
        alpha = alpha * divided["alpha"][time][0]

        matrix.update(divided["translate"][time],scale,rotate)

        box = divided["box"][time][0]
        if(abs(box[2] - box[0])>0):
            cropped = cropImage(img, box)
            cropped = scaleImage(cropped, scale)
            cropped = cropped.rotate(rotate * 180.0 / np.pi, expand=1)

            size = np.array(cropped.size)
            extra = divided["extra"][time][0]
            position = list((center + matrix.getXY() - size / 2 + extra / 2).astype(int))
            #canvas.paste(cropped, position, cropped)
            croppeds.append({
                "img":cropped,
                "pos":position,
                "layer":divided["layer"][time][0],
                "index":index,
                "alpha":alpha,
                "compositeType":divided["compositeType"]
            })

        for childIndex in sorted(divided["children"]):
            child = divides[childIndex]
            drawCropped(child,translate,scale,rotate,alpha,Matrix(matrix),childIndex)

    master = divides[0]
    drawCropped(master, [0, 0], [1, 1], 0, 1, Matrix(), 0)

    srtCropped = sorted(croppeds,key=lambda x:(x["layer"],x["index"]))
    for cropped in srtCropped:
        if cropped["compositeType"] == 0:
            tmpCan = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            R, G, B, A = cropped["img"].split()
            A = np.array(A)
            A = A * cropped["alpha"]
            A = Image.fromarray(A.astype(np.uint8), mode='L')
            tmpImg = Image.merge("RGBA", (R, G, B, A))
            tmpCan.paste(tmpImg, cropped["pos"])
            canvas = Image.alpha_composite(canvas,tmpCan)
        elif cropped["compositeType"] == 2:
            tmpCan = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            tmpCan.paste(cropped["img"], cropped["pos"])
            tmpCan = np.array(tmpCan).astype(int)
            tmpCan2 = np.array(canvas).astype(int)
            tmpCan[:, :, 0] = tmpCan[:, :, 0] * tmpCan[:, :, 3] / 255.0
            tmpCan[:, :, 1] = tmpCan[:, :, 1] * tmpCan[:, :, 3] / 255.0
            tmpCan[:, :, 2] = tmpCan[:, :, 2] * tmpCan[:, :, 3] / 255.0
            tmpCan[:, :, 3] = 0
            tmpCan = tmpCan + tmpCan2
            np.clip(tmpCan, 0, 255, out=tmpCan)
            canvas = Image.fromarray(tmpCan.astype(np.uint8), mode='RGBA')
        elif cropped["compositeType"] == 3:
            tmpCan = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            tmpCan.paste(cropped["img"], cropped["pos"])
            tmpCan = np.array(tmpCan).astype(int)
            tmpCan2 = np.array(canvas).astype(int)
            tmpCan = 255 - tmpCan
            tmpCan[:, :, 0] = tmpCan[:, :, 0] * tmpCan[:, :, 3] / 255.0
            tmpCan[:, :, 1] = tmpCan[:, :, 1] * tmpCan[:, :, 3] / 255.0
            tmpCan[:, :, 2] = tmpCan[:, :, 2] * tmpCan[:, :, 3] / 255.0
            tmpCan[:, :, 3] = 1
            tmpCan = tmpCan * tmpCan2 / 255.0
            np.clip(tmpCan, 0, 255, out=tmpCan)
            canvas = Image.fromarray(tmpCan.astype(np.uint8), mode='RGBA')


    return canvas


with open(fnameSprite,"rb") as fp:
    animPackNum = readInt(fp)
    animPackGuideOffsets = readInt(fp,animPackNum)
    pointer = fp.tell()
    for animPackGuideOffset in animPackGuideOffsets:
        fp.seek(animPackGuideOffset)
        animPackOffset = readInt(fp)*4+pointer+4
        fp.seek(animPackOffset)
        animNum = readInt(fp)
        animOffsets = readInt(fp,animNum)

        for animOffset in animOffsets:
            fp.seek(animOffset+8)
            firstDivided = readInt(fp)
            dividedNum = (firstDivided - (animOffset+8))/4
            fp.seek(animOffset + 8)
            dividedOffsets = readInt(fp,dividedNum)
            divides = {}
            parentIndexs = [0]
            maxTime = 0.0
            while True:
                parentIndex = parentIndexs.pop()
                dividedOffset = dividedOffsets[parentIndex]
                fp.seek(dividedOffset)
                divided,maxTime = parseDivided(fp)
                parentIndexs = parentIndexs + divided["children"]
                divides[parentIndex] = divided

                if len(parentIndexs) == 0:
                    break

            savePath = "%s/%s/%s/" % (unitId,
                        index2DiscrpLut[unitSprType]["animpack"][animPackGuideOffsets.index(animPackGuideOffset)],
                        index2DiscrpLut[unitSprType]["anim"][animOffsets.index(animOffset)])

            for frame in range(int(np.ceil(maxTime*30))+1):
                canvas = drawDivides(divides,frame/30.0)
                os.makedirs(savePath,exist_ok=True)
                canvas.save(savePath + "%d.png" % frame)
