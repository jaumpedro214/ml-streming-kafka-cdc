import confluent_kafka as ck
import pprint
# /connector-plugins/{connectorType}/config/validate
import json

TOPIC = ["car_database.public.car_data"]
if __name__ == "__main__":

    # Read from topic
    consumer = ck.Consumer(
        {
            "bootstrap.servers": "kafka:9092",
            "group.id": "teste",
            "auto.offset.reset": "earliest",
        }
    )

    consumer.subscribe(TOPIC)

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.value() is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue

        # decode message
        message = msg.value().decode("utf-8")
        # load json
        json_message = json.loads(message)

        pprint.pprint(json_message)
        
        # print key if exists
        key = msg.key()
        if key is not None:
            print("Key: {}".format(key))
            
