#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include "helper_cuda.h"

enum DeviceType
{
	CPU,
	GPU,
	UNDEFINED
};

/**
* @brief MaxHeap class
* @details This class is a max heap implementation. It is used to store the k nearest neighbors of a point.
* @tparam T The type of the data stored in the heap
* 
* Heap can be considered as a complete binary tree which is completely filled except the last level.
* The last level is filled from left to right.
* The root element will be at Arr[0].
* Below table shows indexes of other nodes for the ith node, i.e., Arr[i]:
* Arr[(i-1)/2]	Returns the parent node
* Arr[(2*i)+1]	Returns the left child node
* Arr[(2*i)+2]	Returns the right child node
* 
* this heap is used to store the k nearest neighbors of a point. 
* for example, if you have a point cloud with 1000 points, and you want to find the 10 nearest neighbors of a point.
* you can create a MaxHeap with maxSize = 10, and insert the first 10 points into the heap.
* then you can iterate through the remaining points, and check whether the distance between the point 
* and the query point is smaller than the maximum distance in the heap.
* 
*/
template<typename T, DeviceType deviceType> class MaxHeap;

template<typename T>
using MaxHeapCPU = MaxHeap<T, DeviceType::CPU>;

template<typename T>
class MaxHeap<T, DeviceType::CPU>
{
public:
	MaxHeap(const int maxSize);
	~MaxHeap();
	/**
	* @brief Inserts a value into the heap
	* @details Inserts a value into the heap
	* @param value The value to be inserted
	* @param index The index of the value to be inserted
	* 
	* before inserting the value, you shold check whether the value is smaller than the maximum value in the heap.
	* You can use the getMax() function to get the maximum value in the heap. 
	* The heap will not check whether the value is greater than the maximum value.
	*/
	void insert(const T& value, const uint32_t idx);
	void resize(const int maxSize);
	T extractMax();
	T pop()
	{
		return extractMax();
	}
	inline T getMax() const
	{
		if(m_size > 0)
			return m_valueHeap[0];
		else
			return T();
	}

	inline T top() const
	{
		if (m_size > 0)
			return m_valueHeap[0];
		else
			return T();
	}

	inline int size() {
		return m_size;
	}

	inline int maxSize() {
		return m_maxSize;
	}

	inline bool isEmpty() {
		return m_size == 0;
	}

	inline T& operator [] (unsigned int id)
	{
		return m_valueHeap[id];
	}

	inline const T& operator [] (unsigned int id) const
	{
		return m_valueHeap[id];
	}

	inline bool isCPU() const {return true;}
	inline bool isGPU() const {return false;}

	void assign(const MaxHeap<T, DeviceType::GPU>& heap);

	T* valueData() {return m_valueHeap;}
	int* idxData() {return m_indexHeap;}

	friend std::ostream& operator<<(std::ostream& os, const MaxHeap<T, DeviceType::CPU>& heap)
	{
		for (int i = 0; i < heap.m_size; i++)
		{
			os << heap[i] << " ";
		}
		return os;
	}

private:
	int parent(int i) { return (i - 1) / 2; }
	int left(int i) { return 2 * i + 1; }
	int right(int i) { return 2 * i + 2; }

	void heapify(int i);

private:
	T* m_valueHeap;
	int32_t* m_indexHeap;
	int m_maxSize;
	int m_size;
	bool m_is_initialized;
};

template<typename T>
MaxHeap<T, DeviceType::CPU>::MaxHeap(const int maxSize):m_maxSize(maxSize), m_size(0), m_is_initialized(false)
{
	m_valueHeap = new T[maxSize];
	m_indexHeap = new int32_t[maxSize];
	for (int i = 0; i < maxSize; i++)
	{
		m_indexHeap[i] = -1;
	}
	m_is_initialized = true;
}

template<typename T>
MaxHeap<T, DeviceType::CPU>::~MaxHeap()
{
	if (m_is_initialized)
	{
		delete[] m_valueHeap;
		delete[] m_indexHeap;
	}
}


template<typename T>
void MaxHeap<T, DeviceType::CPU>::insert(const T& value, const uint32_t idx)
{
	if (m_size >= m_maxSize)
	{
		extractMax();
	}
	m_valueHeap[m_size] = value;
	m_indexHeap[m_size] = idx;
	int index = m_size;
	m_size++;

	while (index != 0 && m_valueHeap[parent(index)] < m_valueHeap[index])
	{
		std::swap(m_valueHeap[index], m_valueHeap[parent(index)]);
		std::swap(m_indexHeap[index], m_indexHeap[parent(index)]);
		index = parent(index);
	}
}

template<typename T>
inline void MaxHeap<T, DeviceType::CPU>::resize(const int maxSize)
{
	if (m_is_initialized)
	{
		delete[] m_valueHeap;
		delete[] m_indexHeap;
	}
	m_maxSize = maxSize;
	m_size = 0;
	m_valueHeap = new T[maxSize];
	m_indexHeap = new int32_t[maxSize];
	for (int i = 0; i < maxSize; i++)
	{
		m_indexHeap[i] = -1;
	}
	m_is_initialized = true;
}

template<typename T>
T MaxHeap<T, DeviceType::CPU>::extractMax()
{
	if (m_size <= 0)
		return -1;
	T maxValue = m_valueHeap[0];
	m_valueHeap[0] = m_valueHeap[m_size - 1];
	m_indexHeap[0] = m_indexHeap[m_size - 1];
	m_size--;

	heapify(0);
	return maxValue;
}


//template<typename T>
//void MaxHeap<T, DeviceType::CPU>::assign(const MaxHeap<T, DeviceType::GPU>& heap)
//{
//	if (heap.size() != this->m_size())
//		this->resize(heap.size());
//	checkCudaErrors(cudaMemcpy(m_valueHeap, heap.m_valueHeap, sizeof(T) * heap.size(), cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaMemcpy(m_indexHeap, heap.m_indexHeap, sizeof(int32_t) * heap.size(), cudaMemcpyDeviceToHost));
//	this->m_maxSize = heap.m_maxSize;
//}
//

template<typename T>
void MaxHeap<T, DeviceType::CPU>::heapify(int i)
{
	int l = left(i);
	int r = right(i);
	int largest = i;
	if (l < m_size && m_valueHeap[l] > m_valueHeap[i])
		largest = l;
	if (r < m_size && m_valueHeap[r] > m_valueHeap[largest])
		largest = r;
	if (largest != i)
	{
		std::swap(m_valueHeap[i], m_valueHeap[largest]);
		std::swap(m_indexHeap[i], m_indexHeap[largest]);
		heapify(largest);
	}
}
template<typename T>
using MaxHeapCUDA = MaxHeap<T, DeviceType::GPU>;

template<typename T>
class MaxHeap<T, DeviceType::GPU>
{
public:
	MaxHeap(const int maxSize);
	~MaxHeap();
	void deInit();

	void resize(const int maxSize);
	__device__ void insert(const T& value, const int32_t idx);
	__device__ T extractMax();
	__device__ T pop() { return extractMax();}

	__device__ inline T getMax() const
	{
		if (m_size > 0)
			return m_valueHeap[0];
		else
			return T();
	}

	__device__ inline T top() const
	{
		if (m_size > 0)
			return m_valueHeap[0];
		else
			return T();
	}
	__device__ __host__ inline int size(){return m_size;}
	__device__ __host__ inline int maxSize() {return m_maxSize;}
	__device__ __host__ inline bool isEmpty() { return m_size == 0;}
	__device__ __host__ inline bool isCPU() const { return false; }
	__device__ __host__ inline bool isGPU() const { return true; }

	__device__ inline T& operator [] (unsigned int id) { return m_valueHeap[id]; }
	__device__ inline const T& operator [] (unsigned int id) const { return m_valueHeap[id]; }
	__device__ inline T& at(unsigned int id) { return m_valueHeap[id]; }

	T* valueData() { return m_valueHeap; }
	int* idxData() { return m_indexHeap; }

	void getValues(std::vector<T>& values)
	{
		values.resize(m_size);
		checkCudaErrors(cudaMemcpy(values.data(), m_valueHeap, sizeof(T) * m_size, cudaMemcpyDeviceToHost));
	}
	void getIndices(std::vector<int>& indices)
	{
		indices.resize(m_size);
		checkCudaErrors(cudaMemcpy(indices.data(), m_indexHeap, sizeof(int) * m_size, cudaMemcpyDeviceToHost));
	}
	void getData(std::vector<T>& values, std::vector<int>& indices)
	{
		values.resize(m_size);
		indices.resize(m_size);
		checkCudaErrors(cudaMemcpy(values.data(), m_valueHeap, sizeof(T) * m_size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(indices.data(), m_indexHeap, sizeof(int) * m_size, cudaMemcpyDeviceToHost));
	}
private:
	__device__ int parent(int i) { return (i - 1) / 2; }
	__device__ int left(int i) { return 2 * i + 1; }
	__device__ int right(int i) { return 2 * i + 2; }
	__device__ void heapify(int i);

	template<typename Type>
	__device__ void swap(Type& a, Type& b)
	{
		Type temp = a;
		a = b;
		b = temp;
	}
private:
	T* m_valueHeap;
	int* m_indexHeap;
	int m_maxSize;
	int m_size;
	bool m_is_initialized;
};


template<typename T>
MaxHeap<T, DeviceType::GPU>::MaxHeap(const int maxSize) :m_maxSize(maxSize), m_size(0), m_is_initialized(false)
{
	checkCudaErrors(cudaMalloc(&m_valueHeap, sizeof(T) * maxSize));
	checkCudaErrors(cudaMalloc(&m_indexHeap, sizeof(int) * maxSize));
	this->m_is_initialized = true;
}

template<typename T>
void MaxHeap<T, DeviceType::GPU>::deInit()
{
	if (m_is_initialized)
	{
		checkCudaErrors(cudaFree(m_valueHeap));
		checkCudaErrors(cudaFree(m_indexHeap));
		m_is_initialized = false;
	}
}


template<typename T>
MaxHeap<T, DeviceType::GPU>::~MaxHeap()
{
}

template<typename T>
void MaxHeap<T, DeviceType::GPU>::resize(const int maxSize)
{
	if (m_is_initialized)
	{
		checkCudaErrors(cudaFree(this->m_valueHeap));
		checkCudaErrors(cudaFree(this->m_indexHeap));
	}
	m_maxSize = maxSize;
	m_size = 0;
	checkCudaErrors(cudaMalloc(&(this->m_valueHeap), sizeof(T) * maxSize));
	checkCudaErrors(cudaMalloc(&(this->m_indexHeap), sizeof(int) * maxSize));
	m_is_initialized = true;
}

template<typename T>
__device__ void MaxHeap<T, DeviceType::GPU>::insert(const T& value, const int32_t idx)
{
	if (m_size >= m_maxSize)
	{
		extractMax();
	}
	m_valueHeap[m_size] = value;
	m_indexHeap[m_size] = idx;
	int index = m_size;
	m_size++;

	while (index != 0 && m_valueHeap[parent(index)] < m_valueHeap[index])
	{
		this->swap(m_valueHeap[index], m_valueHeap[parent(index)]);
		this->swap(m_indexHeap[index], m_indexHeap[parent(index)]);
		index = parent(index);
	}
}

template<typename T>
__device__ T MaxHeap<T, DeviceType::GPU>::extractMax()
{
	if (m_size <= 0)
		return -1;
	T maxValue = m_valueHeap[0];
	m_valueHeap[0] = m_valueHeap[m_size - 1];
	m_indexHeap[0] = m_indexHeap[m_size - 1];
	m_size--;

	heapify(0);
	return maxValue;
}

template<typename T>
__device__ void MaxHeap<T, DeviceType::GPU>::heapify(int i)
{
	int l = left(i);
	int r = right(i);
	int largest = i;
	if (l < m_size && m_valueHeap[l] > m_valueHeap[i])
		largest = l;
	if (r < m_size && m_valueHeap[r] > m_valueHeap[largest])
		largest = r;
	if (largest != i)
	{
		this->swap(m_valueHeap[i], m_valueHeap[largest]);
		this->swap(m_indexHeap[i], m_indexHeap[largest]);
		heapify(largest);
	}
}
