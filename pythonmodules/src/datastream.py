import datetime
import random
import time
import pandas as pd
import sys
from loguru import logger
import paho.mqtt.client as mqtt
import psycopg2
import json


class Connector:
    """
    Connect mqtt or postgres

    Parameters
    ----------
    connection_type: str
        Connection service, support `mqtt` and `postgres`
    **kwargs
        Required parameters to connect to mqtt or postgres

        For mqtt: transport, url, port, username, password, payload_keys, topic

        For postgers: database, user, password, host, port, query

    Returns
    -------
    None or Dataframe
        - If mqtt, no return.
        - If postgres, return dataframe
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    def on_connect(self, client, userdata, flags, rc):
        """
        The action to be done when the local terminal program connects to the server and gets a response
        """

        print("Connected with result code " + str(rc))

        # If we lose connection or reconnect, the terminal will resubscribe
        client.subscribe(self.params["topic"])

    def on_message(self, client, userdata, msg):
        """
        Action to take when a message sent from the server is received
        """

        print(msg.topic + " " + msg.payload.decode("utf-8"))

    def mqtt_sub(self):
        """
        MQTT Subscriber
        """

        try:
            client = mqtt.Client(transport=self.params["transport"])
            client.on_connect = self.on_connect
            client.on_message = self.on_message
            client.username_pw_set(self.params["username"], self.params["password"])
            client.connect(self.params["url"], self.params["port"], 5)
            client.loop_forever()
        except Exception as e:
            logger.error(e)
            sys.exit(0)

    def mqtt_pub(self):
        """
        MQTT Publisher
        """

        # ISO 8601 format
        ISOTIMEFORMAT = "%Y-%m-%d %H:%M:%S"

        logger.info("Connect mqtt ...")
        print(self.params)
        try:
            client = mqtt.Client(transport=self.params["transport"])
            client.username_pw_set(self.params["username"], self.params["password"])
            client.connect(self.params["url"], self.params["port"], 60)
            payload = {"timestamp": datetime.datetime.now().strftime(ISOTIMEFORMAT)}
            while True:
                for key in self.params["payload_keys"]:
                    payload[key] = round(random.uniform(1, 10), 2)
                print(json.dumps(payload))
                client.publish(self.params["topic"], json.dumps(payload))
                time.sleep(1)
        except Exception as e:
            logger.error(e)
            sys.exit(0)

    def postgres(self):
        """
        Connect to postgersql
        """
        
        try:
            conn = psycopg2.connect(database=self.params["database"], user=self.params["user"],
                                    password=self.params["password"], host=self.params["host"],
                                    port=self.params["port"])
            cur = conn.cursor()
            query = self.params["query"]
            cur.execute(query)
            result = pd.DataFrame(cur.fetchall())
            result.columns = [col.name for col in cur.description]

            conn.commit()
            conn.close()

            return result
        except Exception as e:
            logger.error(e)
            sys.exit(0)
