#include <iostream>
#include <fstream>
#include <strstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/copyleft/marching_cubes.h>

typedef double FT;
typedef Eigen::Matrix<FT, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
typedef Eigen::Matrix<FT, Eigen::Dynamic, 1> VectorX;
typedef Eigen::Matrix<FT, 3, 1> Vector3;

class UniformGrid
{
public:
	UniformGrid()
	: N(0)
	{}
	UniformGrid(int _N) {
		N = _N;
		distances.resize(N);
		for (auto& d : distances) {
			d.resize(N);
			for (auto& v : d)
				v.resize(N, 1e30);
		}
	}
	template <class T>
	T distance(const T* const p) const {
		int px = *(double*)&p[0] * N;
		int py = *(double*)&p[1] * N;
		int pz = *(double*)&p[2] * N;
		if (px < 0 || py < 0 || pz < 0 || px >= N - 1 || py >= N - 1 || pz >= N - 1) {
			T l = (T)0;
			if (px < 0)
				l = l + -p[0] * (T)N;
			else if (px >= N)
				l = l + (p[0] * (T)N - (T)(N - 1 - 1e-3));

			if (py < 0)
				l = l + -p[1] * (T)N;
			else if (py >= N)
				l = l + (p[1] * (T)N - (T)(N - 1 - 1e-3));

			if (pz < 0)
				l = l + -p[2] * (T)N;
			else if (pz >= N)
				l = l + (p[2] * (T)N - (T)(N - 1 - 1e-3));

			return l;
		}
		T wx = p[0] * (T)N - (T)px;
		T wy = p[1] * (T)N - (T)py;
		T wz = p[2] * (T)N - (T)pz;
		T w0 = ((T)1 - wx) * ((T)1 - wy) * ((T)1 - wz) * distances[pz    ][py    ][px    ];
		T w1 = wx 		   * ((T)1 - wy) * ((T)1 - wz) * distances[pz    ][py    ][px + 1];
		T w2 = ((T)1 - wx) * wy 		 * ((T)1 - wz) * distances[pz    ][py + 1][px    ];
		T w3 = wx 		   * wy 		 * ((T)1 - wz) * distances[pz    ][py + 1][px + 1];
		T w4 = ((T)1 - wx) * ((T)1 - wy) * wz 		   * distances[pz + 1][py    ][px    ];
		T w5 = wx 		   * ((T)1 - wy) * wz 		   * distances[pz + 1][py    ][px + 1];
		T w6 = ((T)1 - wx) * wy 		 * wz		   * distances[pz + 1][py + 1][px    ];
		T w7 = wx 		   * wy 		 * wz 		   * distances[pz + 1][py + 1][px + 1];
		T res = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7;
		T thres = (T)0.02;

		//commented out for deform_tune
		if (res > thres)
			res = thres;
		//
		
		return res;
	}
	int N;
	std::vector<std::vector<std::vector<double> > > distances;
};

struct LengthError {
  LengthError(const Eigen::Vector3d& v_, double lambda_)
  : v(v_), lambda(lambda_) {}

  template <typename T>
  bool operator()(const T* const p1,
                  const T* const p2,
                  T* residuals) const {
  	T px = p1[0] - p2[0];
  	T py = p1[1] - p2[1];
  	T pz = p1[2] - p2[2];
  	residuals[0] = px - v[0];
  	residuals[1] = py - v[1];
  	residuals[2] = pz - v[2];
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const Eigen::Vector3d& v, const double lambda_) {
     return (new ceres::AutoDiffCostFunction<LengthError, 3, 3, 3>(
                 new LengthError(v, lambda_)));
   }
   double lambda;
   Eigen::Vector3d v;
};

struct DistanceError {
  DistanceError(UniformGrid* grid_)
  : grid(grid_) {}

  template <typename T>
  bool operator()(const T* const p1,
                  T* residuals) const {
  	residuals[0] = grid->distance(p1);
  	residuals[1] = (T)0;
  	residuals[2] = (T)0;
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(UniformGrid* grid) {
     return (new ceres::AutoDiffCostFunction<DistanceError, 3, 3>(
                 new DistanceError(grid)));
   }
   UniformGrid* grid;
};

class Mesh
{
public:
	Mesh() : scale(1.0) {}
	std::vector<Eigen::Vector3d> V;
	std::vector<Eigen::Vector3i> F;
	void ReadOBJ(const char* filename) {
		std::ifstream is(filename);
		char buffer[256];
		while (is.getline(buffer, 256)) {
			std::strstream str;
			str << buffer;
			str >> buffer;
			if (strcmp(buffer, "v") == 0) {
				double x, y, z;
				str >> x >> y >> z;
				V.push_back(Eigen::Vector3d(x, y, z));
			}
			else if (strcmp(buffer, "f") == 0) {
				Eigen::Vector3i f;
				for (int j = 0; j < 3; ++j) {
					str >> buffer;
					int id = 0;
					int p = 0;
					while (buffer[p] != '/') {
						id = id * 10 + (buffer[p] - '0');
						p += 1;
					}
					f[j] = id - 1;
				}
				F.push_back(f);
			}
		}
	}

	void ReadOBJ_Manifold(const char* filename) {
		std::ifstream is(filename);
		char buffer[256];
		while (is.getline(buffer, 256)) {
			std::strstream str;
			str << buffer;
			str >> buffer;
			if (strcmp(buffer, "v") == 0) {
				double x, y, z;
				str >> x >> y >> z;
				V.push_back(Eigen::Vector3d(x, y, z));
			}
			else if (strcmp(buffer, "f") == 0) {
				Eigen::Vector3i f;
				int idx_x, idx_y, idx_z;
				str >> idx_x >> idx_y >> idx_z;
				// std::cout<<idx_x <<' '<< idx_y <<' '<< idx_z <<std::endl;
				f = Eigen::Vector3i(idx_x-1, idx_y-1, idx_z-1);
				F.push_back(f);
			}
		}
	}

	void WriteOBJ(const char* filename) {
		std::ofstream os(filename);
		for (int i = 0; i < V.size(); ++i) {
			os << "v " << V[i][0] << " " << V[i][1] << " " << V[i][2] << "\n";
		}
		for (int i = 0; i < F.size(); ++i) {
			os << "f " << F[i][0] + 1 << " " << F[i][1] + 1 << " " << F[i][2] + 1 << "\n";
		}
		os.close();
	}
	double scale;
	Eigen::Vector3d pos;
	void Normalize() {
		double min_p[3], max_p[3];
		for (int j = 0; j < 3; ++j) {
			min_p[j] = 1e30;
			max_p[j] = -1e30;
			for (int i = 0; i < V.size(); ++i) {
				if (V[i][j] < min_p[j])
					min_p[j] = V[i][j];
				if (V[i][j] > max_p[j])
					max_p[j] = V[i][j];
			}
		}
		scale = std::max(max_p[0] - min_p[0], std::max(max_p[1] - min_p[1], max_p[2] - min_p[2])) * 1.1;
		for (int j = 0; j < 3; ++j)
			pos[j] = min_p[j] - 0.05 * scale;
		for (auto& v : V) {
			v = (v - pos) / scale;
		}
		for (int j = 0; j < 3; ++j) {
			min_p[j] = 1e30;
			max_p[j] = -1e30;
			for (int i = 0; i < V.size(); ++i) {
				if (V[i][j] < min_p[j])
					min_p[j] = V[i][j];
				if (V[i][j] > max_p[j])
					max_p[j] = V[i][j];
			}
		}
	}
	void ApplyTransform(Mesh& m) {
		pos = m.pos;
		scale = m.scale;
		for (auto& v : V) {
			v = (v - pos) / scale;
		}
	}
	void ConstructDistanceField(UniformGrid& grid) {
		Eigen::MatrixXd P(grid.N * grid.N * grid.N, 3);
		int offset = 0;
		for (int i = 0; i < grid.N; ++i) {
			for (int j = 0; j < grid.N; ++j) {
				for (int k = 0; k < grid.N; ++k) {
					P.row(offset) = Eigen::Vector3d(double(k) / grid.N, double(j) / grid.N, double(i) / grid.N);
					offset += 1;
				}
			}
		}

		Eigen::MatrixXd V2(V.size(), 3);
		for (int i = 0; i < V.size(); ++i)
			V2.row(i) = V[i];

		Eigen::MatrixXi F2(F.size(), 3);
		for (int i = 0; i < F.size(); ++i)
			F2.row(i) = F[i];

		Eigen::MatrixXd N(F.size(), 3);
		for (int i = 0; i < F.size(); ++i) {
			Eigen::Vector3d x = V[F[i][1]] - V[F[i][0]];
			Eigen::Vector3d y = V[F[i][2]] - V[F[i][0]];
			N.row(i) = x.cross(y).normalized();
		}

		Eigen::VectorXd sqrD;
		Eigen::VectorXi I;
		Eigen::MatrixXd C;
		igl::point_mesh_squared_distance(P,V2,F2,sqrD,I,C);

		offset = 0;

		for (int i = 0; i < grid.N; ++i) {
			for (int j = 0; j < grid.N; ++j) {
				for (int k = 0; k < grid.N; ++k) {
					Eigen::Vector3d n = N.row(I[offset]);
					Eigen::Vector3d off = P.row(offset);
					off -= V[F[I[offset]][0]];
					double d = n.dot(off);
					d = 1;
					if (d > 0)
						grid.distances[i][j][k] = sqrt(sqrD[offset]);
					else
						grid.distances[i][j][k] = sqrt(-sqrD[offset]);
					offset += 1;
				}
			}
		}

	}

	void FromDistanceField(UniformGrid& grid) {
		Eigen::VectorXd S(grid.N * grid.N * grid.N);
		Eigen::MatrixXd GV(grid.N * grid.N * grid.N, 3);
		int offset = 0;
		for (int i = 0; i < grid.N; ++i) {
			for (int j = 0; j < grid.N; ++j) {
				for (int k = 0; k < grid.N; ++k) {
					S[offset] = grid.distances[i][j][k];
					GV.row(offset) = Eigen::Vector3d(k, j, i);
					offset += 1;
				}
			}
		}
		Eigen::MatrixXd SV;
		Eigen::MatrixXi SF;
		igl::copyleft::marching_cubes(S,GV,grid.N,grid.N,grid.N,SV,SF);
		V.resize(SV.rows());
		F.resize(SF.rows());
		for (int i = 0; i < SV.rows(); ++i)
			V[i] = SV.row(i) / (double)grid.N;
		for (int i = 0; i < SF.rows(); ++i)
			F[i] = SF.row(i);
	}

	void Deform(UniformGrid& grid) {
		double lambda = 1e-3;
		ceres::Problem problem;

		//Move vertices
		std::vector<ceres::ResidualBlockId> v_block_ids;
		for (int i = 0; i < V.size(); ++i) {
			ceres::CostFunction* cost_function = DistanceError::Create(&grid);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[i].data());
			v_block_ids.push_back(block_id);			
		}

		//Enforce rigidity
		std::vector<ceres::ResidualBlockId> edge_block_ids;
		for (int i = 0; i < F.size(); ++i) {
			for (int j = 0; j < 3; ++j) {
				Eigen::Vector3d v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
				ceres::CostFunction* cost_function = LengthError::Create(v, lambda);
				ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[F[i][j]].data(), V[F[i][(j + 1) % 3]].data());
				edge_block_ids.push_back(block_id);
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << "\n";

		//V error
		ceres::Problem::EvaluateOptions v_options;
		v_options.residual_blocks = v_block_ids;
		double v_cost;
		problem.Evaluate(v_options, &v_cost, NULL, NULL, NULL);
		std::cout<<"Vertices cost: "<<v_cost<<std::endl;

		//E error
		ceres::Problem::EvaluateOptions edge_options;
		edge_options.residual_blocks = edge_block_ids;
		double edge_cost;
		problem.Evaluate(edge_options, &edge_cost, NULL, NULL, NULL);
		std::cout<<"Rigidity cost: "<<edge_cost<<std::endl;

		double final_cost = v_cost + edge_cost;
		std::cout<<"Final cost: "<<final_cost<<std::endl;
	}

	double Get_Final_Cost(Mesh& ref){
		//normalize by number of vertices and faces
		int ref_num_v = ref.V.size();
		int source_num_f = F.size();

		MatrixX SV(V.size(), 3), RV(ref.V.size(), 3);
		Eigen::MatrixXi SF(F.size(), 3);
		for (int i = 0; i < V.size(); ++i)
			SV.row(i) = V[i];
		for (int i = 0; i < ref.V.size(); ++i)
			RV.row(i) = ref.V[i];
		for (int i = 0; i < F.size(); ++i)
			SF.row(i) = F[i];


		VectorX sqrD;
		Eigen::VectorXi I;
		MatrixX C;
		igl::point_mesh_squared_distance(RV, SV, SF,sqrD,I,C);
		FT coverage_cost = sqrD.sum() * 0.5;
		std::cout<<"Coverage cost: "<<coverage_cost << std::endl;

		FT rigidity_cost = 0.0;
		std::cout<<"Rigidity cost: "<<rigidity_cost<<std::endl;

		FT final_cost = coverage_cost/ref_num_v + rigidity_cost/source_num_f;
		std::cout<<"Final cost: "<<final_cost<<std::endl;

		return final_cost;
	}

};

//For deformation
int main(int argc, char** argv) {
	if (argc < 4) {
		printf("./deform source.obj reference.obj output.obj textfile_name\n");
		return 0;
	}
	//Deform source to fit the reference

	Mesh src, ref;
	src.ReadOBJ_Manifold(argv[1]);
	ref.ReadOBJ_Manifold(argv[2]);

	//Get number of vertices and faces
	std::cout<<"Source:\t\t"<<"Num vertices: "<<src.V.size()<<"\tNum faces: "<<src.F.size()<<std::endl;
	std::cout<<"Reference:\t"<<"Num vertices: "<<ref.V.size()<<"\tNum faces: "<<ref.F.size()<<std::endl<<std::endl;

	UniformGrid grid(100);
	ref.Normalize();
	src.ApplyTransform(ref);

	ref.ConstructDistanceField(grid);

	src.Deform(grid);

	double cost;
	cost = src.Get_Final_Cost(ref);

	double threshold = 1e-3;

	//for deform_tune
	// double threshold = 1e-3;
	// double threshold = 5e-4; #does not filter valid deformations

	if (cost > threshold){
		std::cout<<"INVALID"<<std::endl;
		// return 0;
	}

	else {
		std::cout<<"Deformed"<<std::endl;		
	}

	std::string text_file_name = argv[4];
	std::ofstream outfile;
	outfile.open(text_file_name, std::fstream::in | std::fstream::out | std::fstream::app);
	if (!outfile){
		outfile.open(text_file_name,  std::fstream::in | std::fstream::out | std::fstream::trunc);
	} 
	//output_objectcost
	outfile<<argv[3]<<'\t'<<cost<<std::endl;
	outfile.close();

	std::ifstream is(argv[1]);
	std::ofstream os(argv[3]);
	char buffer[1024];
	int offset = 0;
	while (is.getline(buffer, 1024)) {
		if (buffer[0] == 'v' && buffer[1] == ' ') {
			auto v = src.V[offset++] * src.scale + src.pos;
			os << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
		} else {
			os << buffer << "\n";
		}
	}

	is.close();
	os.close();
	return 0;
}

