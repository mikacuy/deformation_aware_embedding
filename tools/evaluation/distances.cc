#include <Eigen/Core>

#include <igl/readOBJ.h>
#include <igl/point_mesh_squared_distance.h>

extern "C" {

Eigen::MatrixXd V1, V2;
Eigen::MatrixXi F1, F2;
void LoadObj(const char* filename, int mesh_id) {
    if (mesh_id == 1)
        igl::readOBJ(filename, V1, F1);
    else
        igl::readOBJ(filename, V2, F2);
}

double OneWayChamferDistance(const Eigen::MatrixXd& V1,
    const Eigen::MatrixXi& F1,
    const Eigen::MatrixXd& V2,
    const Eigen::MatrixXi& F2,
    int resample_points) {
    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    if (resample_points) {
        
        std::vector<double> areas(F1.rows() + 1, 0);
        double total_area = 0;
        for (int i = 0; i < F1.rows(); ++i) {
            const Eigen::Vector3d& v1 = V1.row(F1(i, 0));
            const Eigen::Vector3d& v2 = V1.row(F1(i, 1));
            const Eigen::Vector3d& v3 = V1.row(F1(i, 2));
            double area = ((v2 - v1).cross(v3 - v1)).norm();
            areas[i + 1] = area;
            total_area += area;
        }
        for (int i = 1; i < areas.size(); ++i)
            areas[i] += areas[i - 1];
        
        Eigen::MatrixXd NV(resample_points, 3);
        for (int i = 0; i < resample_points; ++i) {
            double r = rand() / (double)RAND_MAX * areas.back();
            auto lower = std::lower_bound(areas.begin(), areas.end(), r) - 1;
            int tri_idx = lower - areas.begin();
            if (tri_idx < 0) {
                tri_idx = 0;
            }
            double u = rand() / (double)RAND_MAX;
            double v = rand() / (double)RAND_MAX;
            if (u + v > 1) {
                u = 1 - u;
                v = 1 - v;
            }
            auto v1 = V1.row(F1(tri_idx, 0));
            auto v2 = V1.row(F1(tri_idx, 1));
            auto v3 = V1.row(F1(tri_idx, 2));

            NV.row(i) = u * (v2 - v1) + v * (v3 - v1) + v1;
        }
        igl::point_mesh_squared_distance(NV,V2,F2,sqrD,I,C);
    } else {
        igl::point_mesh_squared_distance(V1,V2,F2,sqrD,I,C);
    }
    double distance = 0;
    for (int i = 0; i < sqrD.size(); ++i) {
        distance += sqrt(sqrD[i]);
    }
    double ans = distance / sqrD.size();
    return ans;
}

double OneWayChamfer(int src_id, int resample_points = 0) {
    if (src_id == 1) {
        return OneWayChamferDistance(V1, F1, V2, F2, resample_points);
    }
    return OneWayChamferDistance(V2, F2, V1, F1, resample_points);
}

double TwoWayChamfer(int resample_points = 0) {
    double distance1 = OneWayChamferDistance(V1, F1, V2, F2, resample_points);
    double distance2 = OneWayChamferDistance(V2, F2, V1, F1, resample_points);
    
    return distance1 + distance2;
}

void SetMesh(double* vertices, int* faces, int num_V, int num_F, int id) {
    auto &V = (id == 1) ? V1 : V2;
    auto &F = (id == 1) ? F1 : F2;

    V = Eigen::MatrixXd(num_V, 3);
    for (int i = 0; i < num_V; ++i) {
        V.row(i) = Eigen::Vector3d(vertices[i * 3],vertices[i * 3 + 1],vertices[i * 3 + 2]);
    }

    F = Eigen::MatrixXi(num_F, 3);
    for (int i = 0; i < num_F; ++i) {
        F.row(i) = Eigen::Vector3i(faces[i * 3], faces[i * 3 + 1], faces[i * 3 + 2]);
    }
}

double AreaRatio(const char* f1, const char* f2) {
    LoadObj(f1, 2);
    LoadObj(f2, 1);
    {
        Eigen::MatrixXd V2_buf(V2.rows() * 2, 3);
        Eigen::MatrixXi F2_buf(F2.rows() * 2, 3);
        for (int i = 0; i < V2.rows(); ++i) {
            V2_buf.row(i) = V2.row(i);
            V2_buf.row(i + V2.rows()) = V2.row(i);
            V2_buf(i + V2.rows(), 0) = -V2_buf(i + V2.rows(), 0);
        }
        for (int i = 0; i < F2.rows(); ++i) {
            F2_buf.row(i) = F2.row(i);
            F2_buf.row(i + F2.rows()) = F2.row(i);
            for (int j = 0; j < 3; ++j)
                F2_buf(i + F2.rows(), j) += V2.rows();
        }
        F2 = F2_buf;
        V2 = V2_buf;
    }        
    {
        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXd C;
        std::vector<double> areas(F1.rows() + 1, 0);
        double total_area = 0;
        for (int i = 0; i < F1.rows(); ++i) {
            const Eigen::Vector3d& v1 = V1.row(F1(i, 0));
            const Eigen::Vector3d& v2 = V1.row(F1(i, 1));
            const Eigen::Vector3d& v3 = V1.row(F1(i, 2));
            double area = ((v2 - v1).cross(v3 - v1)).norm();
            areas[i + 1] = area;
            total_area += area;
        }
        for (int i = 1; i < areas.size(); ++i)
            areas[i] += areas[i - 1];
        
        int resample_points = 100000;
        Eigen::MatrixXd NV(resample_points, 3);
        for (int i = 0; i < resample_points; ++i) {
            double r = rand() / (double)RAND_MAX * areas.back();
            auto lower = std::lower_bound(areas.begin(), areas.end(), r) - 1;
            int tri_idx = lower - areas.begin();
            if (tri_idx < 0) {
                tri_idx = 0;
            }
            double u = rand() / (double)RAND_MAX;
            double v = rand() / (double)RAND_MAX;
            if (u + v > 1) {
                u = 1 - u;
                v = 1 - v;
            }
            auto v1 = V1.row(F1(tri_idx, 0));
            auto v2 = V1.row(F1(tri_idx, 1));
            auto v3 = V1.row(F1(tri_idx, 2));

            NV.row(i) = u * (v2 - v1) + v * (v3 - v1) + v1;
        }

        igl::point_mesh_squared_distance(NV,V2,F2,sqrD,I,C);
        int inliers = 0;
        for (int i = 0; i < sqrD.size(); ++i) {
            double dis = sqrt(sqrD[i]);
            if (dis < 0.03)
                inliers += 1;
        }
        return (double)inliers / sqrD.size();
    }
}

};