import psycopg2

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="nombreDB3",      # Nombre de la base de datos
            user="postgres3",        # Usuario maestro
            password="Contra1234*",  # Contraseña maestra
            host="database3.cm6jksktwdcb.us-east-1.rds.amazonaws.com",  # Endpoint
            port="5432"              # Puerto
        )
        return conn
    except Exception as e:
        print("Error al conectar a la base de datos:", e)
        return None
