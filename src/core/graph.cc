#include "core/graph.h"
#include "core/op_type.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>    
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        //removetarget即为从tensor中删除一个以该tensor作为输入的算子
        //removesource即为删除生成该tensor的算子
        auto ops_size = ops.size();
        for (size_t i = 0; i < ops_size; i++)
        {
            auto op = ops[i];
            if (op->getOpType() == OpType::Transpose){
                auto op_transpose = std::dynamic_pointer_cast<TransposeObj>(op);
                auto input = op_transpose->getInputs(0);
                auto preOp = input->getSource();
                // 0 1 3 2
                // 0 1 3 2

                if (preOp && preOp->getOpType() == OpType::Transpose && input->targets.size() == 1){
                    auto preOp_transpose = std::dynamic_pointer_cast<TransposeObj>(preOp);
                    auto preInput = preOp_transpose->getInputs(0);
                    auto perm_op1 = op_transpose->getPermute();
                    auto perm_op2 = preOp_transpose->getPermute();
                    bool merge_flag = true;
                    for (size_t m = 0; m < perm_op1.size(); m++){
                        perm_op1[m] = perm_op2[perm_op1[m]];
                        if (perm_op1[m] != int(m)){
                            merge_flag = false;
                        }
                    }
                    preInput->removeTarget(preOp);
                    if (merge_flag){
                        for (auto succ : op->getSuccessors()){
                            succ->replaceInput(op->getOutput(), preInput);
                            preInput->addTarget(succ);
                        }
                        this->removeTensor(op->getOutput());
                    }
                    else{
                        auto new_op = make_ref<TransposeObj>(this, preInput, op->getOutput(), perm_op1);
                        this->addOperatorAndConnect(new_op);
                    }
                    //即对于preOp的前驱算子，删除preOp
                    for (auto pre : preOp->getPredecessors()){
                        pre->removeSuccessors(preOp);
                    }
                    //即对于op的后继算子，删除op
                    for (auto suc : op->getSuccessors()){
                        suc->removePredecessors(op);
                    }
                    ops_size -= 2;
                    i -= 2;
                    this->removeOperator(op);
                    this->removeOperator(preOp);
                    this->removeTensor(input);
                    continue;
                }
            }
        if (op->getOpType() == OpType::MatMul){
            auto op_mul = std::dynamic_pointer_cast<MatmulObj>(op);
            auto mata = op_mul->getInputs(0);
            auto matb = op_mul->getInputs(1);
            auto pre_op_mul_a = mata->getSource();
            auto pre_op_mul_b = matb->getSource();
            // mat的输入经过了transpose算子的操作，且只有这一个算子作用
            if (pre_op_mul_a && pre_op_mul_a->getOpType() == OpType::Transpose && mata->targets.size() == 1){
                auto op_transpose_a = std::dynamic_pointer_cast<TransposeObj>(pre_op_mul_a);
                auto perm = op_transpose_a->getPermute();
                bool merge_flag = true;
                //判断前面的维度是否相同
                for (size_t m = 0; m < perm.size() - 2; m++){
                    if (perm[m] != int(m)){
                        merge_flag = false;
                        break;
                    }
                }
                //判断最后两个维度是否相同 并且后两个维度交换
                if (!merge_flag || perm[perm.size() - 1] != int(perm.size() - 2) || perm[perm.size() - 2] != int(perm.size() - 1)){
                    continue;
                }
                auto transpose_input = pre_op_mul_a->getInputs(0);
                op_mul->setTransA(!op_mul->getTransA());
                //删除前驱算子
                op_mul->removePredecessors(pre_op_mul_a);
                for (auto pre : pre_op_mul_a->getPredecessors()){
                    //对于transpose算子的前驱算子，删除transpose算子并更换为matmul
                    //对于matmul算子的前驱算子更换为transpose算子的前驱算子
                    pre->removeSuccessors(pre_op_mul_a);
                    pre->addSuccessors(op_mul);
                    op->addPredecessors(pre);
                }
                transpose_input->removeTarget(pre_op_mul_a);
                transpose_input->addTarget(op_mul);
                op_mul->inputs[0] = transpose_input;
                this->removeTensor(mata);
                this->removeOperator(pre_op_mul_a);
                ops_size--;
                i--;
                //删除中间tensor
            }
            if (pre_op_mul_b && pre_op_mul_b->getOpType() == OpType::Transpose && matb->targets.size() == 1){
                auto op_transpose_b = std::dynamic_pointer_cast<TransposeObj>(pre_op_mul_b);
                auto perm = op_transpose_b->getPermute();
                bool merge_flag = true;
                for (size_t m = 0; m < perm.size() - 2; m++){
                    if (perm[m] != int(m)){
                        merge_flag = false;
                        break;
                    }
                }
                if (!merge_flag || perm[perm.size() - 1] != int(perm.size() - 2) || perm[perm.size() - 2] != int(perm.size() - 1)){
                    continue;
                }
                auto transpose_input = pre_op_mul_b->getInputs(0);
                op_mul->setTransB(!op_mul->getTransB());
                op_mul->removePredecessors(pre_op_mul_b);
                for (auto pre : pre_op_mul_b->getPredecessors()){
                    pre->removeSuccessors(pre_op_mul_b);
                    pre->addSuccessors(op_mul);
                    op->addPredecessors(pre);
                }
                transpose_input->removeTarget(pre_op_mul_b);
                transpose_input->addTarget(op_mul);
                op_mul->inputs[1] = transpose_input;
                this->removeTensor(matb);
                this->removeOperator(pre_op_mul_b);
                ops_size--;
                i--;
            }
        }
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc() {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业
        // ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给
        // tensor 绑定内存
        // =================================== 作业
        // ===================================
        auto n = this->tensors.size();
        vector<size_t> offsets(n);
        for (size_t i = 0; i < n; i++) {
            offsets[i] = this->allocator.alloc(this->tensors[i]->getBytes());
        }
        auto hptr = this->allocator.getPtr();
        for (size_t i = 0; i < n; i++) {
            auto ptr = hptr + offsets[i];
            auto blob = make_ref<BlobObj>(this->runtime, ptr);
            this->tensors[i]->setDataBlob(blob);
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini