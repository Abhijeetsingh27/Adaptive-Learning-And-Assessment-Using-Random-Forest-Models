import MySQLdb

try:
    db = MySQLdb.connect(host="localhost", user="root", passwd="")
    cursor = db.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS plrs_db")
    print("Database created successfully")
except Exception as e:
    print(f"Failed to create database: {e}")
finally:
    try:
        db.close()
    except:
        pass
