import cv2, os, random, colorsys, onnxruntime, string, time, argparse, uuid, logging, itertools
import numpy as np


parser = argparse.ArgumentParser("ocr-")
parser.add_argument('-i',"--input", type = str, required = True, default = False, help = "path image ...")
logging.basicConfig(filename=f'log/ocr.log', filemode='w', format='%(asctime)s - %(message)s', level = logging.INFO, datefmt='%d-%b-%y %H:%M:%S')


class Process(object):
	def __init__(self):
		pass


	def normalize(self, img: np.ndarray) -> np.ndarray:
	    IMG_H = 64
	    IMG_W = 128
	    if img.shape[-1] == 3:
	        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	    img = cv2.resize(img, (IMG_W, IMG_H))

	    img = img.astype(np.float32)
	    img -= np.amin(img)
	    img /= np.amax(img)
	    img = [[[h] for h in w] for w in img.T]

	    x = np.zeros((IMG_W, IMG_H, 1))
	    x[:, :, :] = img
	    x = x[np.newaxis]
	    return x 


	def decode_batch(self, out: np.ndarray, letters:list):
	    ret = []
	    for j in range(out.shape[0]):
	        out_best = list(np.argmax(out[j, 2:], 1))
	        out_best = [k for k, g in itertools.groupby(out_best)]
	        outstr = ''
	        for c in out_best:
	            if c < len(letters):
	                outstr += letters[c]
	        ret.append(outstr)
	    return ret


class OCR(Process):
	def __init__(self, path_model:str):
		self.session = onnxruntime.InferenceSession(path_model)
		self.letters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','E','H','I','K','M','O','P','T','X']


	def inference(self, input_1):
	    input_1 = np.float32(input_1)
	    ort_inputs = {self.session.get_inputs()[0].name:input_1}
	    ocr = self.session.run(None, ort_inputs)
	    return ocr[0]


	def __call__(self, image_path:str) -> str:
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = self.normalize(image)
		out   = self.inference(image)
		out   = self.decode_batch(out, self.letters)
		return out[0]


if __name__ == '__main__':
	args = parser.parse_args()
	ocr = OCR(path_model = "ocr-ua.onnx")
	pred = ocr(args.input)
	logging.info(f'name image: {args.input} ocr: {pred}')