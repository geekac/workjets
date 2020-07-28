#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#define _CRT_SECURE_NO_WARNINGS		// for mbstowcs

#include <iostream>
#include <vector>

#include "Python.h"
#include "numpy/arrayobject.h"

// 第三方开源库libCZI头文件
#include "inc_libCZI.h"

using namespace std;
typedef unsigned char uint8;
typedef unsigned int  uint32_t;



// https://stackoverflow.com/questions/3342726/c-print-out-enum-value-as-text
std::ostream& operator<<(std::ostream& out, const libCZI::PixelType value) {
	static std::map<libCZI::PixelType, std::string> strings;
	if (strings.size() == 0) {
#define INSERT_ELEMENT(p) strings[p] = #p
		INSERT_ELEMENT(libCZI::PixelType::Invalid);
		INSERT_ELEMENT(libCZI::PixelType::Gray8);
		INSERT_ELEMENT(libCZI::PixelType::Gray16);
		INSERT_ELEMENT(libCZI::PixelType::Gray32Float);
		INSERT_ELEMENT(libCZI::PixelType::Bgr24);
		INSERT_ELEMENT(libCZI::PixelType::Bgr48);
		INSERT_ELEMENT(libCZI::PixelType::Bgr96Float);
		INSERT_ELEMENT(libCZI::PixelType::Bgra32);
		INSERT_ELEMENT(libCZI::PixelType::Gray64ComplexFloat);
		INSERT_ELEMENT(libCZI::PixelType::Bgr192ComplexFloat);
		INSERT_ELEMENT(libCZI::PixelType::Gray32);
		INSERT_ELEMENT(libCZI::PixelType::Gray64Float);
#undef INSERT_ELEMENT
	}

	return out << strings[value];
}

std::ostream& operator<<(std::ostream& out, const libCZI::DimensionIndex value) {
	static std::map<libCZI::DimensionIndex, std::string> strings;
	if (strings.size() == 0) {
#define INSERT_ELEMENT(p) strings[p] = #p
		INSERT_ELEMENT(libCZI::DimensionIndex::invalid);
		INSERT_ELEMENT(libCZI::DimensionIndex::MinDim);
		INSERT_ELEMENT(libCZI::DimensionIndex::Z);
		INSERT_ELEMENT(libCZI::DimensionIndex::C);
		INSERT_ELEMENT(libCZI::DimensionIndex::T);
		INSERT_ELEMENT(libCZI::DimensionIndex::R);
		INSERT_ELEMENT(libCZI::DimensionIndex::S);
		INSERT_ELEMENT(libCZI::DimensionIndex::I);
		INSERT_ELEMENT(libCZI::DimensionIndex::H);
		INSERT_ELEMENT(libCZI::DimensionIndex::V);
		INSERT_ELEMENT(libCZI::DimensionIndex::B);
		INSERT_ELEMENT(libCZI::DimensionIndex::MaxDim);
#undef INSERT_ELEMENT
	}

	return out << strings[value];
}


/* #### Globals #################################### */

// .... Python callable extensions .................. //
static PyObject *cnt_subblocks(PyObject *self, PyObject *args);
static PyObject *cziread_subblock(PyObject *self, PyObject *args);



/* ==== Set up the methods table ====================== */
static PyMethodDef _czifileMethods[] = {
	{"cnt_subblocks", cnt_subblocks, METH_VARARGS, "return number of total subblocks"},
	{"cziread_subblock", cziread_subblock, METH_VARARGS, "Read czi image subblock at `idx`"},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};


// https://docs.python.org/3.6/extending/extending.html
// http://python3porting.com/cextensions.html

static struct PyModuleDef moduledef = {
		PyModuleDef_HEAD_INIT,
		"_czifile",          /* m_name */
		NULL,                /* m_doc */
		-1,                  /* m_size */
		_czifileMethods,     /* m_methods */
		NULL,                /* m_reload */
		NULL,                /* m_traverse */
		NULL,                /* m_clear */
		NULL                 /* m_free */
};


// generic exception for any errors encountered here
static PyObject *CzifileError;

extern "C" {
	PyMODINIT_FUNC PyInit__czifile(void)
	{
		PyObject *module = PyModule_Create(&moduledef);

		if (module == NULL)
			return NULL;

		CzifileError = PyErr_NewException("czifile.error", NULL, NULL);
		Py_INCREF(CzifileError);
		PyModule_AddObject(module, "_czifile_exception", CzifileError);

		import_array();  // 使用Numpy之前必须执行这行代码！！

		return module;
	}
}



/* #### Helper prototypes ################################### */

std::shared_ptr<libCZI::ICZIReader> open_czireader_from_cfilename(char const *fn);
PyArrayObject* copy_bitmap_to_numpy_array(std::shared_ptr<libCZI::IBitmapData> pBitmap);



/* #### Extended modules #################################### */

static PyObject *cnt_subblocks(PyObject *self, PyObject *args) {
	char *filename_buf;
	// parse arguments
	if (!PyArg_ParseTuple(args, "s", &filename_buf))
		return NULL;

	auto cziReader = open_czireader_from_cfilename(filename_buf);
	/*
		auto sbStatistics = cziReader->GetStatistics();
		int subblock_count = sbStatistics.subBlockCount
		这个counting here all sub-block (no matter on which pyramid-layer).
		不准确
	*/

	 //count all the subblocks
	int subblock_count = 0;
	cziReader->EnumerateSubBlocks(
		[&subblock_count](int idx, const libCZI::SubBlockInfo& info)
	{
		subblock_count++;
		return true;
	});
	return Py_BuildValue("i", subblock_count);
}


static PyObject *cziread_subblock(PyObject *self, PyObject *args) {
	/* 这个函数读取指定idx subblock的信息：
			logicalRect_x  :  逻辑x坐标
			logicalRect_y  :  逻辑y坐标    这两个数值一组
			physicalSize_h :  高度
			physicalSize_w :  宽度		  这两个数值一组  --- 因而一共返回四个  需要四个变量接收返回值
			zoom           :  zoom值 放缩
			PyArray        :  图像的像素值 np.ndarrag类型 shape=(高度*宽度*通道数, )
	*/
	int logicalRect_x(0), logicalRect_y(0);
	uint32_t physicalSize_h(0), physicalSize_w(0);
	double zoom(0.0);

	char *filename_buf;
	int block_idx = 0;
	// parse arguments
	if (!PyArg_ParseTuple(args, "si", &filename_buf, &block_idx))
		return NULL;

	auto cziReader = open_czireader_from_cfilename(filename_buf);
	auto subBlk = cziReader->ReadSubBlock(block_idx);

	//获取subblock信息
	libCZI::SubBlockInfo info = subBlk->GetSubBlockInfo();
	logicalRect_x = info.logicalRect.x;
	logicalRect_y = info.logicalRect.y;
	physicalSize_h = info.physicalSize.h;
	physicalSize_w = info.physicalSize.w;
	cout << info.logicalRect.h << info.logicalRect.w << info.physicalSize << endl;
	zoom = info.GetZoom();

	//获取subblock的np.ndarray数据：
	auto bm = subBlk->CreateBitmap();
	std::uint32_t height = bm->GetHeight();
	std::uint32_t width = bm->GetWidth();

	int numpy_type = NPY_UINT16; int pixel_size_bytes = 0; int channels = 1;
	switch (bm->GetPixelType()) {
	case libCZI::PixelType::Gray8:
		numpy_type = NPY_UINT8; pixel_size_bytes = 1; channels = 1;
		break;
	case libCZI::PixelType::Gray16:
		numpy_type = NPY_UINT16; pixel_size_bytes = 2; channels = 1;
		break;
	case libCZI::PixelType::Bgr48:
		numpy_type = NPY_UINT16; pixel_size_bytes = 6; channels = 3;
		break;

		// Add by achange:  支持我需要处理的czi文件类型
	case libCZI::PixelType::Bgr24:
		numpy_type = NPY_UINT8; pixel_size_bytes = 1; channels = 3;
		break;

	default:
		std::cout << bm->GetPixelType() << std::endl;
		PyErr_SetString(CzifileError, "Unknown image type in czi file, ask to add more types.");
		return NULL;
	}

	npy_intp eshp[1]; eshp[0] = height*width*channels;	// 返回图像数组shape=(h*w*c, )
	PyArrayObject *subblock0 = (PyArrayObject *)PyArray_Empty(1, eshp, PyArray_DescrFromType(NPY_UINT8), 0);

	// 开始读取数据  并写入到np数组
	auto bitmap = bm->Lock();
	uint8 *cimgptr = (uint8*)bitmap.ptrDataRoi;

	void *pointer = PyArray_DATA(subblock0);
	npy_byte *cptr = (npy_byte*)pointer;
	std::memcpy(cptr, cimgptr, height*width*channels);

	return Py_BuildValue("(i,i)(I,I)dO",
		logicalRect_x, logicalRect_y, physicalSize_h, physicalSize_w, zoom, (PyObject *)subblock0
	);
}


PyArrayObject* copy_bitmap_to_numpy_array(std::shared_ptr<libCZI::IBitmapData> pBitmap) {
	// define numpy types/shapes and bytes per pixel depending on the zeiss bitmap pixel type.
	int numpy_type = NPY_UINT16; int pixel_size_bytes = 0; int channels = 1;
	switch (pBitmap->GetPixelType()) {
	case libCZI::PixelType::Gray8:
		numpy_type = NPY_UINT8; pixel_size_bytes = 1; channels = 1;
		break;
	case libCZI::PixelType::Gray16:
		numpy_type = NPY_UINT16; pixel_size_bytes = 2; channels = 1;
		break;
	case libCZI::PixelType::Bgr48:
		numpy_type = NPY_UINT16; pixel_size_bytes = 6; channels = 3;
		break;

		// Add by achange:  支持我需要处理的czi文件类型
	case libCZI::PixelType::Bgr24:
		numpy_type = NPY_UINT8; pixel_size_bytes = 1; channels = 3;
		break;

	default:
		std::cout << pBitmap->GetPixelType() << std::endl;
		PyErr_SetString(CzifileError, "Unknown image type in czi file, ask to add more types.");
		return NULL;
	}

	// allocate the numpy matrix to copy image into
	//cout << size_x << " " << size_y << endl;
	auto size = pBitmap->GetSize();
	int size_x = size.w, size_y = size.h;
	PyArrayObject *img;
	int swap_axes[2];
	if (channels == 1) {
		npy_intp shp[2]; shp[0] = size_x; shp[1] = size_y;
		// images in czi file are in F-order, set F-order flag (last argument to PyArray_Empty)
		img = (PyArrayObject *)PyArray_Empty(2, shp, PyArray_DescrFromType(numpy_type), 1);
		swap_axes[0] = 0; swap_axes[1] = 1;
	}
	else {
		npy_intp shp[3]; shp[1] = size_x; shp[2] = size_y; shp[0] = channels;
		// images in czi file are in F-order, set F-order flag (last argument to PyArray_Empty)
		img = (PyArrayObject *)PyArray_Empty(3, shp, PyArray_DescrFromType(numpy_type), 1);
		swap_axes[0] = 0; swap_axes[1] = 2;
	}
	void *pointer = PyArray_DATA(img);

	// copy from the czi lib image pointer to the numpy array pointer
	auto bitmap = pBitmap->Lock();
	//cout << "sixe_x " << size_x << " size y " << size_y << endl;
	//cout << "stride " << bitmap.stride << " size " << bitmap.size << endl;
	// can not do a single memcpy call because the stride does not necessarily match the row size.
	npy_byte *cptr = (npy_byte*)pointer, *cimgptr = (npy_byte*)bitmap.ptrDataRoi;
	// stride units is not documented but emperically means the row (x) stride in bytes, not in pixels.
	int rowsize = pixel_size_bytes * size_x; //, imgrowsize = pixel_size_bytes * bitmap.stride;
	for (int y = 0; y < size_y; y++) {
		std::memcpy(cptr, cimgptr, rowsize);
		cptr += rowsize; cimgptr += bitmap.stride;
	}
	pBitmap->Unlock();

	// transpose to convert from F-order to C-order array
	return (PyArrayObject*)PyArray_SwapAxes(img, swap_axes[0], swap_axes[1]);
}





std::shared_ptr<libCZI::ICZIReader> open_czireader_from_cfilename(char const *fn) {
	// open the czi file
	// https://msdn.microsoft.com/en-us/library/ms235631.aspx
	size_t newsize = strlen(fn) + 1;
	// The following creates a buffer large enough to contain
	// the exact number of characters in the original string
	// in the new format. If you want to add more characters
	// to the end of the string, increase the value of newsize
	// to increase the size of the buffer.
	wchar_t * wcstring = new wchar_t[newsize];
	// Convert char* string to a wchar_t* string.
	//size_t convertedChars = mbstowcs(wcstring, fn, newsize);
	mbstowcs(wcstring, fn, newsize);
	auto cziReader = libCZI::CreateCZIReader();
	auto stream = libCZI::CreateStreamFromFile(wcstring);
	delete[] wcstring;
	cziReader->Open(stream);

	return cziReader;
}


