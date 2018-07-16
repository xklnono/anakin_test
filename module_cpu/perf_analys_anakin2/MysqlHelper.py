#!/usr/bin/env python
#encoding:utf-8

import MySQLdb
class MysqlHelper(object):
    """数据库工具类"""
    def __init__(self, host=None, port=None, user=None, passwd=None, db=None):    
        """init with None"""
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset = None
        self.connection = None
    def __del__(self):
        """make sure close the connection"""
        self.close()
    def connect(self):
        """connect to DB"""
        if self.connection is not None:
            self.connection.close()
        try:
            self.connection = MySQLdb.connect(host=self.host, port=self.port,\
            user=self.user, passwd=self.passwd, db=self.db, charset='utf8')
        except Exception as e:
            print e
            self.connection = None
            raise MyException
        
    def close(self):
        """close connection if conected"""
        if (self.connection is not None):
            self.connection.close()
            self.connection = None
    def execute(self, sql):
        """execute a sql command without return value"""
        if (self.connection is  None):
            self.connect()
        cursor = self.connection.cursor()
        
        try:
            cursor.execute(sql)
            self.connection.commit()
        except Exception as e:
            print("Execute sql command '%s': failed: %s" % (sql, e))
            self.connection.rollback()
            raise MyException
        finally:
            cursor.close() 
    def executes(self, sql):
        """execute a sql command and return the result"""
        if (self.connection is None):
            self.connect()
        cursor = self.connection.cursor()
        
        try:
            cursor.execute(sql)
            self.connection.commit()
            result = cursor.fetchall()
        except Exception as  e:
            print("Execute sql command '%s': failed: %s" % (sql, e))
            self.connection.rollback()
            raise MyException
        finally:
            cursor.close() 
            
        return result
    def executeone(self, sql):
        """execute a sql command and return the result"""
        if (self.connection is None):
            self.connect()
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            self.connection.commit()
            result = cursor.fetchone()
        except Exception as  e:
            print("Execute sql command '%s': failed: %s" % (sql, e))
            self.connection.rollback()
            raise MyException
        finally:
            cursor.close()
        return result
    def executemany(self, sql, params=None):
        """execute more thon one command and return the result"""
        if (self.connection is None):
            self.connect()
        cursor = self.connection.cursor()
        try:
            cursor.executemany(sql, params)
            self.connection.commit()
        except Exception as  e:
            print("Execute sql command '%s': failed: %s" % (sql, e))
            self.connection.rollback()
        finally:
            cursor.close()

    def execute_withnodb(self, sql):
        """execute a sql command without db"""
        try:
            connection_nodb = MySQLdb.connect(host=self.host, port=self.port,\
                user=self.user, passwd=self.passwd, charset='utf8')
        except Exception as e:
            print e
            self.connection_nodb = None
            raise MyException

        cursor = connection_nodb.cursor()

        try:
            cursor.execute(sql)
            connection_nodb.commit()
        except Exception as e:
            print("Execute sql command '%s': failed: %s" % (sql, e))
            connection_nodb.rollback()
            raise MyException
        finally:
            cursor.close()
