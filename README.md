# Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training [ICCV 2025]

**Official PyTorch implementation of [*Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training*] [ICCV 2025].**

**Wooseong Jeong\***, **Jegyeong Cho\***, **Youngho Yoon\***, and **Kuk-Jin Yoon**  
*Visual Intelligence Lab., KAIST, Korea*  
📧 stk14570@kaist.ac.kr, j2k0618@kaist.ac.kr, dudgh1732@kaist.ac.kr, kjyoon@kaist.ac.kr  


Wooseong Jeong & Kuk-Jin Yoon, Korea Advanced Institute of Science and Technology (KAIST)

Generalizing neural networks to unseen target domains is a significant challenge in real-world deployments. Test-time training (TTT) addresses this by using an auxiliary self-supervised task to reduce the domain gap caused by distribution shifts between the source and target. However, we find that when models are required to perform multiple tasks under domain shifts, conventional TTT methods suffer from unsynchronized task behavior, where the adaptation steps needed for optimal performance in one task may not align with the requirements of other tasks. To address this, we propose a novel TTT approach called Synchronizing Tasks for Test-time Training (S4T), which enables the concurrent handling of multiple tasks. The core idea behind S4T is that predicting task relations across domain shifts is key to synchronizing tasks during test time. To validate our approach, we apply S4T to conventional multi-task benchmarks, integrating it with traditional TTT protocols. Our empirical results show that S4T outperforms state-of-the-art TTT methods across various benchmarks.

## We are preparing the code for public release. It will be available here soon.

## Contact
Wooseong Jeong: stk14570@kaist.ac.kr
