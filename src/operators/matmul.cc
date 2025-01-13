#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto M = inputs[0];
        auto N = inputs[1];
        auto M_shape = M->getDims();
        auto N_shape = N->getDims();
        if (this->transA)
        {
            std::swap(M_shape[M_shape.size() - 1], M_shape[M_shape.size() - 2]);
        }
        if (this->transB)
        {
            std::swap(N_shape[N_shape.size() - 1], N_shape[N_shape.size() - 2]);
        }
        auto output_shape = M_shape;
        output_shape[output_shape.size() - 1] = N_shape[N_shape.size() - 1];
        return {{output_shape}};
        
    }

} // namespace infini