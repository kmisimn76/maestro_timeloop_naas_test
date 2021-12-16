# !!!!! TO FIX K, C, Cluster (line 9, 10, 16)

Network MyTest {
	Layer conv {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 32, C: 32, R: 3, S: 3, Y:7, X:7 }
		Dataflow {
			TemporalMap(8,8) K;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(Y),1) Y;
			TemporalMap(Sz(X),1) X;	
			SpatialMap(1,1) K;
			Cluster(8, P);
			SpatialMap(1,1) C;
		}
	}
}

