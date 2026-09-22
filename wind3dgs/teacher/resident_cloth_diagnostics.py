"""천 정밀도 진단의 작은 GPU 기록. 기본 천 실행에는 연결하지 않는다."""
import warp as wp


@wp.kernel
def store(control:wp.array(dtype=wp.int32),adaptive:wp.array(dtype=wp.int32),
          failure:wp.array(dtype=wp.int32),energy:wp.array(dtype=wp.float64),
          linear:wp.array(dtype=wp.float64),counts:wp.array2d(dtype=wp.int32),
          values:wp.array2d(dtype=wp.float64),slot:int):
    i=wp.tid()
    if i<17:counts[slot,i]=control[i]
    elif i<27:counts[slot,i]=adaptive[i-17]
    elif i==27:counts[slot,i]=failure[1]
    else:counts[slot,i]=failure[0]
    if i<3:values[slot,i]=energy[i]
    elif i==3:values[slot,i]=linear[0]
    elif i==4:values[slot,i]=linear[1]
    elif i==5:values[slot,i]=linear[5]
