import requests, time
def run():
    seq = [[1,0.05*i,0.05*i,0,0,0] for i in range(20)]
    print(requests.post('http://127.0.0.1:8080/predict/anomaly', json={'sequence': seq}).json())
    print(requests.post('http://127.0.0.1:8080/panic', json={'tourist_id':1,'lat':28.6,'lon':77.2}).json())
if __name__ == '__main__':
    run()


