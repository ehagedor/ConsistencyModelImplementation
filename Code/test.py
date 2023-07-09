import torch



def test(tf):
    tf = tf + 1
    print(tf)

#a = test(tf=5)

def test2():
    batch = torch.ones([5,3,3])
    microbatch = 2
    i = 1
    last_batch = (i + microbatch) >= batch.shape[0]
    print(last_batch)

def test3():
    rate = [
            torch.rand([1,1])
            for _ in range(4)
            ]
    print(len(rate))


class Test:
    def __init__(self, ema_rate):
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        
def test4():
    a = torch.randn([768])
    b = torch.randn([768, 192])
    c = torch.add(a,b)
    print(c.shape)
test4()