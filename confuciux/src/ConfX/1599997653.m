Constant KTileSz 4;
Constant CTileSz 1;
Constant ClusterSz 128;
Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 128, C: 128, Y: 112, X: 112, R: 3, S: 3 }
Dataflow {

			TemporalMap(KTileSz,KTileSz) K;
			SpatialMap(Sz(R),Sz(R)) Y';
			TemporalMap(ClusterSz,ClusterSz) X';
            TemporalMap(CTileSz,CTileSz) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			Cluster(ClusterSz, P);
			TemporalMap(1,1) C;
			TemporalMap(1,1) Y';
			SpatialMap(1,1) X';
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}

}
}