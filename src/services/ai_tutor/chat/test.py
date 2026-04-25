from langchain_core.runnables import RunnableLambda, RunnableSequence
from chains.escalation_check import ESCALATION_CHECK_CHAIN


runnable1 = RunnableLambda(lambda x: x + 1)
runnable2 = RunnableLambda(lambda x: x + 2)

#создаем цепочку из двух обьект RunnableLambda
chain = RunnableSequence(runnable1, runnable2)

#вызывваем цепочку
print(chain.invoke(2))
