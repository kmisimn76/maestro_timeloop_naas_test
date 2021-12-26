# !!!!! TO FIX K, C, Cluster (line 9, 10, 16)

Network MyTest {
	Layer conv {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N:1, K: 64, C: 64, R: 3, S: 3, Y:7, X:7 }
		Dataflow {
			TemporalMap(1,1) N;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			Cluster(32, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;	
			SpatialMap(1,1) C;
		}
	}
}

